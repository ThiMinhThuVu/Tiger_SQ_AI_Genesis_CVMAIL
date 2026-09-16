#!/usr/bin/env python3
"""MedSAM ViT-B image-encoder + semantic decoder baseline."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/MedSAM")); sys.path.insert(0, str(ROOT))
from segment_anything import sam_model_registry
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import binary_dice, normalized_hausdorff

def args():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--fold',type=int,default=0)
    p.add_argument('--run-all-folds',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--data-root',type=Path,default=ROOT/'data'); p.add_argument('--output-root',type=Path,default=ROOT/'artifacts/new_baselines'); return p.parse_args()
def seed(v): random.seed(v); np.random.seed(v); torch.manual_seed(v)
class MedSAMSegmenter(nn.Module):
    def __init__(self, checkpoint, labels, height, width, freeze=True):
        super().__init__(); self.height=height; self.width=width
        sam=sam_model_registry['vit_b'](checkpoint=checkpoint); self.encoder=sam.image_encoder
        if freeze:
            for p in self.encoder.parameters(): p.requires_grad=False
        self.decoder=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),nn.Conv2d(128,labels,1))
    def forward(self, image):
        mean=torch.tensor([123.675,116.28,103.53],device=image.device).view(1,3,1,1)
        std=torch.tensor([58.395,57.12,57.375],device=image.device).view(1,3,1,1)
        rgb=(image*torch.tensor([0.229,0.224,0.225],device=image.device).view(1,3,1,1)+torch.tensor([0.485,0.456,0.406],device=image.device).view(1,3,1,1))*255
        rgb=F.interpolate(rgb,size=(1024,1024),mode='bilinear',align_corners=False)
        feat=self.encoder((rgb-mean)/std)
        return F.interpolate(self.decoder(feat),size=image.shape[-2:],mode='bilinear',align_corners=False)
def dice_loss(logits,target):
    p=logits.softmax(1); y=F.one_hot(target,logits.shape[1]).permute(0,3,1,2).float(); inter=(p*y).sum((0,2,3)); den=p.sum((0,2,3))+y.sum((0,2,3)); return 1-((2*inter+1e-5)/(den+1e-5)).mean()
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); losses=[]; ds=[]; hs=[]; correct=pixels=0
    for b in loader:
        t=b['fine'].to(device); z=model(b['image'].to(device)); losses.append(float((F.cross_entropy(z,t)+dice_loss(z,t)).cpu())); p=z.argmax(1).cpu().numpy(); y=t.cpu().numpy(); correct+=int((p==y).sum()); pixels+=y.size
        for a,c in zip(p,y): ds.append(binary_dice(a!=0,c!=0)); hs.append(normalized_hausdorff(a!=0,c!=0))
    return {'loss':float(np.mean(losses)),'fine_dice':float(np.mean(ds)),'fine_nhd':float(np.mean(hs)),'fine_pixel_accuracy':correct/pixels}
def run(config,fold,ns):
    if not torch.cuda.is_available() and not ns.allow_cpu: raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed(int(config['seed'])+fold); lm=LabelMap.load(ns.data_root/'labelmap.csv')
    tr=TigerMultitaskDataset(records_for_fold(ns.data_root,fold,'train'),lm,config['height'],config['width'],True); va=TigerMultitaskDataset(records_for_fold(ns.data_root,fold,'validation'),lm,config['height'],config['width'],False)
    kw={'num_workers':config['num_workers'],'pin_memory':device.type=='cuda'}; tl=DataLoader(tr,batch_size=1,shuffle=True,**kw); vl=DataLoader(va,batch_size=1,shuffle=False,**kw)
    model=MedSAMSegmenter(config['checkpoint'],config['num_labels'],config['height'],config['width'],config['freeze_encoder']).to(device); opt=torch.optim.AdamW(model.decoder.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']))
    out=ns.output_root/config['experiment_id']/f'fold_{fold}'; out.mkdir(parents=True,exist_ok=True); best=1e9; hist=[]; epochs=1 if ns.smoke else int(config['epochs'])
    for ep in range(epochs):
        model.train(); runloss=[]; opt.zero_grad(set_to_none=True)
        for step,b in enumerate(tl):
            t=b['fine'].to(device); z=model(b['image'].to(device)); loss=F.cross_entropy(z,t)+dice_loss(z,t); (loss/int(config['gradient_accumulation'])).backward(); runloss.append(float(loss.detach().cpu()))
            if (step+1)%int(config['gradient_accumulation'])==0 or step+1==len(tl): opt.step(); opt.zero_grad(set_to_none=True)
        m=evaluate(model,vl,device); rec={'epoch':ep,'train_loss':float(np.mean(runloss)),**{f'val_{k}':v for k,v in m.items()}}; hist.append(rec); print(json.dumps(rec),flush=True)
        if m['loss']<best: best=m['loss']; torch.save({'model':model.state_dict(),'config':config,'metrics':rec},out/'best.pt')
    (out/'epoch_metrics.json').write_text(json.dumps(hist,indent=2)); (out/'completion.json').write_text(json.dumps({'status':'smoke' if ns.smoke else 'completed','fold':fold,'epochs':epochs},indent=2))
def main():
    ns=args(); c=yaml.safe_load(ns.config.read_text()); folds=range(5) if ns.run_all_folds else [0 if ns.smoke else ns.fold]
    for f in folds: run(c,f,ns)
if __name__=='__main__': main()
