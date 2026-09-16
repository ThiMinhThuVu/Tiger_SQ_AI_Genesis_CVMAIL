#!/usr/bin/env python3
"""Aggregate 5-fold test metrics for DeepLabV3 or ConvNeXt-V2 Mask2Former."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiger_models.data import LabelMap,TigerMultitaskDataset,records_for_fold,STATIONS
from tiger_models.metrics import segmentation_metrics,visibility_metrics,selection_score
from scripts.train_convnextv2_mask2former import collate,semantic_logits

def build(kind,cfg):
    if kind=='deeplabv3':
        from scripts.train_deeplabv3 import DeepLabV3Multitask
        return DeepLabV3Multitask(False,cfg['fine_classes'])
    from scripts.convnextv2_mask2former import build_model
    return build_model(cfg['model_name'],False,cfg['num_labels'],cfg['decoder_layers'],cfg['num_queries'])

@torch.no_grad()
def fold_metrics(kind,cfg,model_dir,data_root,fold,device,lm):
    path=model_dir/f'fold_{fold}'/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);model=build(kind,cfg).to(device);model.load_state_dict(ck['model'],strict=True);model.eval();ds=TigerMultitaskDataset(records_for_fold(data_root,fold,'test'),lm,cfg['height'],cfg['width'],False);loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate if kind=='mask2former' else None);fp=[];ft=[];cases=[];vt=[];vp=[]
    for b in loader:
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            if kind=='deeplabv3':o=model(b['image'].to(device));logits=o['fine_logits'];vis=o['visibility_logits']
            else:o,vis=model(b['image'].to(device));logits=F.interpolate(semantic_logits(o).float(),size=b['fine'].shape[-2:],mode='bilinear',align_corners=False)
        fp.extend(logits.argmax(1).cpu().numpy());ft.extend(b['fine'].numpy());cases.extend(b['case_id']);vt.append(b['visibility'].numpy());vp.append(vis.float().sigmoid().cpu().numpy())
    fine=segmentation_metrics(fp,ft,cases,lm.fine_weights,lm.fine_names);cp=[lm.fine_to_coarse[x] for x in fp];ct=[lm.fine_to_coarse[x] for x in ft];coarse=segmentation_metrics(cp,ct,cases,lm.coarse_weights,lm.coarse_names);v=visibility_metrics(np.concatenate(vt),np.concatenate(vp),STATIONS);score=selection_score(fine['dice'],fine['nhd'],coarse['dice'],coarse['nhd'],v['macro_f1'],v['macro_auroc']);return {'fold':fold,'checkpoint_epoch':ck['epoch'],'fine_dice':fine['dice'],'fine_nhd':fine['nhd'],'coarse_dice':coarse['dice'],'coarse_nhd':coarse['nhd'],'visibility_f1':v['macro_f1'],'visibility_auroc':v['macro_auroc'],'selection_score':score}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['deeplabv3','mask2former'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=yaml.safe_load(a.config.read_text());lm=LabelMap.load(a.data_root/'labelmap.csv');rows=[fold_metrics(a.model,cfg,a.model_dir,a.data_root,f,torch.device('cuda'),lm) for f in range(5)];names=('fine_dice','fine_nhd','coarse_dice','coarse_nhd','visibility_f1','visibility_auroc','selection_score');summary={k:{'mean':float(np.mean([r[k] for r in rows])),'sample_standard_deviation':float(np.std([r[k] for r in rows],ddof=1))} for k in names};payload={'experiment':cfg['experiment_id'],'completed_test_folds':5,'folds':rows,'summary':summary};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
