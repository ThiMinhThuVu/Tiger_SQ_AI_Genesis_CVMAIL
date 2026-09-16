#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, yaml
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_mask2former import build_model, collate, semantic_logits
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import segmentation_metrics

COLORS=np.asarray([[25,25,25],[255,210,0],[230,30,30],[30,90,230],[220,30,220]],dtype=np.uint8)
def boundary(x):
    b=np.zeros(x.shape,bool); b[1:]|=x[1:]!=x[:-1]; b[:-1]|=x[:-1]!=x[1:]; b[:,1:]|=x[:,1:]!=x[:,:-1]; b[:,:-1]|=x[:,:-1]!=x[:,1:]; return b
def error_map(t,p):
    c=np.zeros(t.shape,np.uint8); d=t!=p; c[d&(t!=0)&(p!=0)]=1; c[(t==0)&(p!=0)]=2; c[(t!=0)&(p==0)]=3; c[boundary(t)^boundary(p)]=4; return COLORS[c]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--split',default='test'); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,default=ROOT/'data'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    cfg=yaml.safe_load(a.config.read_text()); lm=LabelMap.load(a.data_root/'labelmap.csv'); model=build_model(cfg).cuda(); model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'],strict=False); model.eval()
    ds=TigerMultitaskDataset(records_for_fold(a.data_root,a.fold,a.split),lm,cfg['height'],cfg['width'],training=False); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate); rows=[]
    with torch.no_grad():
        for b in loader:
            o=model(pixel_values=b['image'].cuda()); z=torch.nn.functional.interpolate(semantic_logits(o),size=b['fine'].shape[-2:],mode='bilinear',align_corners=False); pred=z.argmax(1).cpu().numpy()[0]; target=b['fine'].numpy()[0]; m=segmentation_metrics([pred],[target],[b['case_id'][0]],lm.fine_weights,lm.fine_names); rows.append((b['name'][0],pred,target,m['dice'],m['nhd']))
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); rows.sort(key=lambda x:x[0]); selected={f"frame_{i:02d}":row for i,row in enumerate(rows)}; manifest={}
    for cat,(name,pred,target,dice,nhd) in selected.items():
        rec=next(r for r in records_for_fold(a.data_root,a.fold,a.split) if r.name==name); image=np.asarray(Image.open(rec.image).convert('RGB')); tc=lm.fine_to_coarse[target]; pc=lm.fine_to_coarse[pred]; fig,ax=plt.subplots(2,5,figsize=(24,10)); ax=ax.ravel(); ims=[image,lm.encode_fine(target),lm.encode_fine(pred),error_map(target,pred),image,lm.encode_coarse(tc),lm.encode_coarse(pc),error_map(tc,pc),image,None]; titles=['Original','Fine GT','Fine prediction','Fine error','Boundaries GT/pred','Coarse GT','Coarse prediction','Coarse error','Original reference',''];
        for i,(im,title) in enumerate(zip(ims,titles)):
            if im is not None: ax[i].imshow(im)
            if i==4: ax[i].contour(boundary(target),levels=[.5],colors='lime'); ax[i].contour(boundary(pred),levels=[.5],colors='red',linestyles='dashed')
            ax[i].set_title(title); ax[i].axis('off')
        fig.suptitle(f'Mask2Former {cat}: {name} | Fine Dice={dice:.4f}, NHD={nhd:.4f}'); fig.tight_layout(); path=out/f'{cat}_{Path(name).stem}.png'; fig.savefig(path,dpi=140); plt.close(fig); manifest[cat]={'name':name,'dice':dice,'nhd':nhd,'figure':path.name}
    (out/'selections.json').write_text(json.dumps(manifest,indent=2)+'\n'); lines=[f'# Mask2Former qualitative — fold {a.fold} ({a.split})',''];
    for cat,x in manifest.items(): lines += [f'## {cat.title()}','','Frame `{}`; Fine Dice `{:.4f}`; NHD `{:.4f}`.'.format(x['name'],x['dice'],x['nhd']),'',f"![{cat}]({x['figure']})",'']
    (out/'MASK2FORMER_QUALITATIVE.md').write_text('\n'.join(lines))
if __name__=='__main__': main()
