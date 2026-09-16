#!/usr/bin/env python3
"""Generate qualitative panels for the best available ConvNeXt-V2 fold."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, torch, yaml
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.convnextv2_common import ConvNeXtV2Multitask
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold, STATIONS
from tiger_models.metrics import segmentation_metrics, visibility_metrics

def boundary(x):
    b=np.zeros(x.shape,bool); b[1:]|=x[1:]!=x[:-1]; b[:-1]|=x[:-1]!=x[1:]; b[:,1:]|=x[:,1:]!=x[:,:-1]; b[:,:-1]|=x[:,:-1]!=x[:,1:]; return b
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--split',default='test'); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,default=ROOT/'data'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Qualitative inference requires CUDA')
    cfg=yaml.safe_load(a.config.read_text()); lm=LabelMap.load(a.data_root/'labelmap.csv'); model=ConvNeXtV2Multitask(cfg['model_name'],cfg['fine_classes'],cfg['visibility_classes'],False,cfg['decoder_channels']).cuda(); model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']); model.eval(); records=records_for_fold(a.data_root,a.fold,a.split); ds=TigerMultitaskDataset(records,lm,cfg['height'],cfg['width'],False); rows=[]
    with torch.no_grad():
        for b in DataLoader(ds,batch_size=1,shuffle=False,num_workers=2):
            o=model(b['image'].cuda()); pred=o['fine_logits'].argmax(1).cpu().numpy()[0]; gt=b['fine'].numpy()[0]; vm=visibility_metrics(b['visibility'].numpy(),o['visibility_logits'].sigmoid().cpu().numpy(),STATIONS); sm=segmentation_metrics([pred],[gt],[b['case_id'][0]],lm.fine_weights,lm.fine_names); rows.append({'name':b['name'][0],'pred':pred,'gt':gt,'prob':o['visibility_logits'].sigmoid().cpu().numpy()[0],'dice':sm['dice'],'nhd':sm['nhd'],'vis_errors':int(((o['visibility_logits'].sigmoid().cpu().numpy()[0]>=.5).astype(int)!=b['visibility'].numpy()[0]).sum())})
    selected={'best':max(rows,key=lambda x:x['dice']),'median':sorted(rows,key=lambda x:x['dice'])[len(rows)//2],'worst':min(rows,key=lambda x:x['dice']),'highest_nhd':max(rows,key=lambda x:x['nhd']),'many_task3_errors':max(rows,key=lambda x:x['vis_errors'])}; a.output_dir.mkdir(parents=True,exist_ok=True); manifest={}
    for cat,r in selected.items():
        rec=next(x for x in records if x.name==r['name']); image=np.asarray(Image.open(rec.image).convert('RGB')); coarse_gt=lm.fine_to_coarse[r['gt']]; coarse_pred=lm.fine_to_coarse[r['pred']]; fig,ax=plt.subplots(2,5,figsize=(24,10)); ax=ax.ravel(); panels=[image,lm.encode_fine(r['gt']),lm.encode_fine(r['pred']),image,lm.encode_coarse(coarse_gt),lm.encode_coarse(coarse_pred),image,image,image,None]; titles=['Original','Fine GT','Fine prediction','Boundaries GT/pred','Coarse GT','Coarse prediction','Fine error','Coarse error','Task 3 probability',''];
        for i,(im,t) in enumerate(zip(panels,titles)):
            if im is not None: ax[i].imshow(im)
            if i==3: ax[i].contour(boundary(r['gt']),levels=[.5],colors='lime'); ax[i].contour(boundary(r['pred']),levels=[.5],colors='red',linestyles='dashed')
            if i==6: ax[i].imshow(np.where(r['gt']==r['pred'],0,1),cmap='magma')
            if i==7: ax[i].imshow(np.where(coarse_gt==coarse_pred,0,1),cmap='magma')
            if i==8: ax[i].barh(np.arange(14),r['prob']); ax[i].set_yticks(np.arange(14),STATIONS); ax[i].set_xlim(0,1); ax[i].invert_yaxis(); ax[i].axvline(.5,color='black',ls='--')
            ax[i].set_title(t); ax[i].axis('off')
        fig.suptitle(f'ConvNeXt-V2 {cat}: {r["name"]} | Fine Dice={r["dice"]:.4f}, nHD={r["nhd"]:.4f}, Task3 errors={r["vis_errors"]}'); fig.tight_layout(); path=a.output_dir/f'{cat}_{Path(r["name"]).stem}.png'; fig.savefig(path,dpi=130); plt.close(fig); manifest[cat]={'name':r['name'],'dice':r['dice'],'nhd':r['nhd'],'visibility_errors':r['vis_errors'],'figure':path.name}
    (a.output_dir/'selections.json').write_text(json.dumps(manifest,indent=2)+'\n'); (a.output_dir/'CONVNEXTV2_QUALITATIVE.md').write_text('\n'.join([f'# ConvNeXt-V2 qualitative — fold {a.fold} ({a.split})','']+[f'## {k.title()}\n\nFrame `{v["name"]}`; fine Dice `{v["dice"]:.4f}`; nHD `{v["nhd"]:.4f}`; Task 3 errors `{v["visibility_errors"]}`.\n\n![{k}]({v["figure"]})\n' for k,v in manifest.items()]))
if __name__=='__main__': main()
