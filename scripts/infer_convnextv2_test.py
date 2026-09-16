#!/usr/bin/env python3
"""Five-fold quantitative test evaluation for ConvNeXt-V2 multitask runs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.convnextv2_common import ConvNeXtV2Multitask
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import segmentation_metrics, visibility_metrics, selection_score
from tiger_models.data import STATIONS

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,default=ROOT/'data'); p.add_argument('--output',type=Path,required=True); p.add_argument('--allow-incomplete',action='store_true'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Test inference requires CUDA')
    cfg=yaml.safe_load(a.config.read_text()); lm=LabelMap.load(a.data_root/'labelmap.csv'); results=[]; device=torch.device('cuda')
    for fold in range(5):
        path=a.model_dir/f'fold_{fold}'/'best.pt'
        if not path.is_file(): continue
        ck=torch.load(path,map_location='cpu',weights_only=False); model=ConvNeXtV2Multitask(cfg['model_name'],cfg['fine_classes'],cfg['visibility_classes'],False,cfg['decoder_channels']).to(device); model.load_state_dict(ck['model']); model.eval()
        ds=TigerMultitaskDataset(records_for_fold(a.data_root,fold,'test'),lm,cfg['height'],cfg['width'],False); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2); fp=[]; ft=[]; cases=[]; vt=[]; vp=[]; correct=total=0
        with torch.no_grad():
            for b in loader:
                o=model(b['image'].to(device)); pred=o['fine_logits'].argmax(1).cpu().numpy()[0]; target=b['fine'].numpy()[0]; fp.append(pred); ft.append(target); cases.append(b['case_id'][0]); vt.append(b['visibility'].numpy()[0]); vp.append(o['visibility_logits'].sigmoid().cpu().numpy()[0]); correct+=int((pred==target).sum()); total+=target.size
        fine=segmentation_metrics(fp,ft,cases,lm.fine_weights,lm.fine_names); cp=[lm.fine_to_coarse[x] for x in fp]; ct=[lm.fine_to_coarse[x] for x in ft]; coarse=segmentation_metrics(cp,ct,cases,lm.coarse_weights,lm.coarse_names); vis=visibility_metrics(np.stack(vt),np.stack(vp),STATIONS)
        results.append({'fold':fold,'checkpoint':str(path),'checkpoint_epoch':ck.get('epoch'),'fine_dice':fine['dice'],'fine_nhd':fine['nhd'],'coarse_dice':coarse['dice'],'coarse_nhd':coarse['nhd'],'visibility_f1':vis['macro_f1'],'visibility_auroc':vis['macro_auroc'],'selection_score':selection_score(fine['dice'],fine['nhd'],coarse['dice'],coarse['nhd'],vis['macro_f1'],vis['macro_auroc']),'fine_pixel_accuracy':correct/total,'visibility_per_station':vis['per_station']})
    if len(results)<5 and not a.allow_incomplete: raise RuntimeError(f'Only {len(results)}/5 checkpoints available')
    names=('fine_dice','fine_nhd','coarse_dice','coarse_nhd','visibility_f1','visibility_auroc','selection_score','fine_pixel_accuracy'); summary={n:{'mean':float(np.mean([r[n] for r in results])),'sample_standard_deviation':float(np.std([r[n] for r in results],ddof=1)) if len(results)>1 else 0.0} for n in names}
    out={'experiment':cfg['experiment_id'],'model_family':'convnextv2','completed_test_folds':len(results),'folds':results,'summary':summary,'note':'Task 1 fine segmentation; Task 2 coarse prediction derived from verified fine-to-coarse mapping; Task 3 14-label visibility.'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
