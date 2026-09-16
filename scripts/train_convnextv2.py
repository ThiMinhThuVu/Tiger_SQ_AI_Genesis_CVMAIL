#!/usr/bin/env python3
"""Train the ConvNeXt-V2 TIGER three-task baseline with case-level CV."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.convnextv2_common import ConvNeXtV2Multitask, multitask_loss
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold, visibility_pos_weight
from tiger_models.metrics import segmentation_metrics, visibility_metrics, selection_score

def args():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--fold',type=int,default=0); p.add_argument('--run-all-folds',action='store_true')
    p.add_argument('--smoke',action='store_true'); p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--data-root',type=Path,default=ROOT/'data'); p.add_argument('--output-root',type=Path,default=ROOT/'artifacts/new_baselines')
    return p.parse_args()

def seed(v): random.seed(v); np.random.seed(v); torch.manual_seed(v)

@torch.no_grad()
def evaluate(model, loader, device, lm):
    model.eval(); fine_p=[]; fine_t=[]; cases=[]; vis_t=[]; vis_p=[]; losses=[]
    for b in loader:
        out=model(b['image'].to(device)); losses.append(float(F.cross_entropy(out['fine_logits'],b['fine'].to(device))))
        fine_p.extend(out['fine_logits'].argmax(1).cpu().numpy()); fine_t.extend(b['fine'].numpy()); cases.extend(b['case_id'])
        vis_t.append(b['visibility'].numpy()); vis_p.append(out['visibility_logits'].sigmoid().cpu().numpy())
    fine=segmentation_metrics(fine_p,fine_t,cases,lm.fine_weights,lm.fine_names)
    cp=[lm.fine_to_coarse[x] for x in fine_p]; ct=[lm.fine_to_coarse[x] for x in fine_t]
    coarse=segmentation_metrics(cp,ct,cases,lm.coarse_weights,lm.coarse_names)
    vis=visibility_metrics(np.concatenate(vis_t),np.concatenate(vis_p),['6L','6R','7L','7R','8','9','10L','10R','11L','11R','12L','12R','13L','13R'])
    score=selection_score(fine['dice'],fine['nhd'],coarse['dice'],coarse['nhd'],vis['macro_f1'],vis['macro_auroc'])
    return {'loss':float(np.mean(losses)),'fine_dice':fine['dice'],'fine_nhd':fine['nhd'],'coarse_dice':coarse['dice'],'coarse_nhd':coarse['nhd'],'visibility_f1':vis['macro_f1'],'visibility_auroc':vis['macro_auroc'],'selection_score':score}

def run_fold(cfg, fold, ns):
    if not torch.cuda.is_available() and not ns.allow_cpu: raise RuntimeError('CUDA unavailable; use --allow-cpu only for smoke')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed(int(cfg['seed'])+fold); lm=LabelMap.load(ns.data_root/'labelmap.csv')
    train_records=records_for_fold(ns.data_root,fold,'train'); val_records=records_for_fold(ns.data_root,fold,'validation')
    train=TigerMultitaskDataset(train_records,lm,cfg['height'],cfg['width'],training=True); val=TigerMultitaskDataset(val_records,lm,cfg['height'],cfg['width'],training=False)
    kw={'num_workers':int(cfg['num_workers']),'pin_memory':device.type=='cuda'}
    tr=DataLoader(train,batch_size=int(cfg['batch_size']),shuffle=True,**kw); va=DataLoader(val,batch_size=1,shuffle=False,**kw)
    model=ConvNeXtV2Multitask(cfg['model_name'],cfg['fine_classes'],cfg['visibility_classes'],cfg['pretrained'],cfg['decoder_channels']).to(device)
    enc=[p for p in model.encoder.parameters() if p.requires_grad]; dec=list(model.lateral.parameters())+list(model.fuse.parameters())+list(model.fine_head.parameters())+list(model.visibility_head.parameters())
    opt=torch.optim.AdamW([{'params':enc,'lr':float(cfg['encoder_learning_rate'])},{'params':dec,'lr':float(cfg['learning_rate'])}],weight_decay=float(cfg['weight_decay']))
    pos=torch.from_numpy(visibility_pos_weight(train_records)).to(device); epochs=1 if ns.smoke else int(cfg['epochs']); out=ns.output_root/cfg['experiment_id']/f'fold_{fold}'; out.mkdir(parents=True,exist_ok=True); best=-float('inf'); history=[]
    for epoch in range(epochs):
        model.train(); running=[]; opt.zero_grad(set_to_none=True)
        for step,b in enumerate(tr):
            batch={k:v.to(device) for k,v in b.items() if isinstance(v,torch.Tensor)}; losses=multitask_loss(model(batch['image']),batch,pos); (losses['total']/int(cfg['gradient_accumulation'])).backward(); running.append(float(losses['total'].detach().cpu()))
            if (step+1)%int(cfg['gradient_accumulation'])==0 or step+1==len(tr): torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        m=evaluate(model,va,device,lm); rec={'epoch':epoch,'train_loss':float(np.mean(running)),**{f'val_{k}':v for k,v in m.items()}}; history.append(rec); print(json.dumps(rec),flush=True)
        if m['selection_score'] is not None and m['selection_score']>best: best=m['selection_score']; torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch,'metrics':rec},out/'best.pt')
    (out/'epoch_metrics.json').write_text(json.dumps(history,indent=2)); (out/'completion.json').write_text(json.dumps({'status':'smoke' if ns.smoke else 'completed','fold':fold,'epochs':epochs},indent=2))

def main():
    ns=args(); cfg=yaml.safe_load(ns.config.read_text()); folds=range(5) if ns.run_all_folds else [0 if ns.smoke else ns.fold]
    for fold in folds: run_fold(cfg,fold,ns)
if __name__=='__main__': main()
