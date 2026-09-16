#!/usr/bin/env python3
"""Torchvision DeepLabV3-ResNet50 three-task TIGER baseline."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tiger_models.data import LabelMap,TigerMultitaskDataset,records_for_fold,visibility_pos_weight,STATIONS
from tiger_models.metrics import segmentation_metrics,visibility_metrics,selection_score

def replace_batchnorm(module: nn.Module) -> None:
    """Make DeepLab stable for the required physical batch size of one."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(32, child.num_features)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features,
                                                eps=child.eps, affine=child.affine))
        else:
            replace_batchnorm(child)

class DeepLabV3Multitask(nn.Module):
    def __init__(self,pretrained=True,fine_classes=31):
        super().__init__(); from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50
        # Load ImageNet weights into the encoder while keeping a 31-class
        # challenge-specific segmentation head.
        self.model=deeplabv3_resnet50(weights=None,weights_backbone=ResNet50_Weights.DEFAULT if pretrained else None,num_classes=fine_classes)
        replace_batchnorm(self.model.classifier)
        self.visibility_head=nn.Sequential(nn.Linear(2048,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,14))
    def forward(self,x):
        features=self.model.backbone(x); logits=self.model.classifier(features['out']); logits=F.interpolate(logits,size=x.shape[-2:],mode='bilinear',align_corners=False); pooled=F.adaptive_avg_pool2d(features['out'],1).flatten(1)
        return {'fine_logits':logits,'visibility_logits':self.visibility_head(pooled)}

def loss(out,b,pos):
    ce=F.cross_entropy(out['fine_logits'],b['fine'].long()); p=out['fine_logits'].softmax(1); hot=F.one_hot(b['fine'].long(),31).permute(0,3,1,2).float(); inter=(p*hot).sum((0,2,3)); den=p.sum((0,2,3))+hot.sum((0,2,3)); dice=1-((2*inter+1e-5)/(den+1e-5)).mean(); vis=F.binary_cross_entropy_with_logits(out['visibility_logits'],b['visibility'].float(),pos_weight=pos); return ce+dice+vis
@torch.no_grad()
def evaluate(model,loader,device,lm):
    model.eval(); fp=[];ft=[];cases=[];vt=[];vp=[]
    for b in loader:
        o=model(b['image'].to(device));fp.extend(o['fine_logits'].argmax(1).cpu().numpy());ft.extend(b['fine'].numpy());cases.extend(b['case_id']);vt.append(b['visibility'].numpy());vp.append(o['visibility_logits'].sigmoid().cpu().numpy())
    f=segmentation_metrics(fp,ft,cases,lm.fine_weights,lm.fine_names);cp=[lm.fine_to_coarse[x] for x in fp];ct=[lm.fine_to_coarse[x] for x in ft];c=segmentation_metrics(cp,ct,cases,lm.coarse_weights,lm.coarse_names);v=visibility_metrics(np.concatenate(vt),np.concatenate(vp),STATIONS);s=selection_score(f['dice'],f['nhd'],c['dice'],c['nhd'],v['macro_f1'],v['macro_auroc']);return {'fine_dice':f['dice'],'fine_nhd':f['nhd'],'coarse_dice':c['dice'],'coarse_nhd':c['nhd'],'visibility_f1':v['macro_f1'],'visibility_auroc':v['macro_auroc'],'selection_score':s}
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--run-all-folds',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true');p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output-root',type=Path,default=ROOT/'artifacts/new_baselines');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
    if not torch.cuda.is_available() and not a.allow_cpu: raise RuntimeError('CUDA unavailable; use --allow-cpu only for smoke')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); folds=range(5) if a.run_all_folds else [a.fold]
    for fold in folds:
        random.seed(cfg['seed']+fold);np.random.seed(cfg['seed']+fold);torch.manual_seed(cfg['seed']+fold);lm=LabelMap.load(a.data_root/'labelmap.csv');trds=TigerMultitaskDataset(records_for_fold(a.data_root,fold,'train'),lm,cfg['height'],cfg['width'],True);vlds=TigerMultitaskDataset(records_for_fold(a.data_root,fold,'validation'),lm,cfg['height'],cfg['width'],False);kw={'num_workers':cfg['num_workers'],'pin_memory':dev.type=='cuda'};tr=DataLoader(trds,batch_size=cfg['batch_size'],shuffle=True,**kw);va=DataLoader(vlds,batch_size=1,shuffle=False,**kw);m=DeepLabV3Multitask(cfg['pretrained'],cfg['fine_classes']).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);pos=torch.from_numpy(visibility_pos_weight(records_for_fold(a.data_root,fold,'train'))).to(dev);out=a.output_root/cfg['experiment_id']/f'fold_{fold}';out.mkdir(parents=True,exist_ok=True);best=-1;history=[];epochs=1 if a.smoke else cfg['epochs']
        for e in range(epochs):
            m.train();opt.zero_grad(set_to_none=True);running=[]
            for i,b in enumerate(tr):
                tb={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)};z=loss(m(tb['image']),tb,pos);(z/cfg['gradient_accumulation']).backward();running.append(float(z.detach().cpu()))
                if (i+1)%cfg['gradient_accumulation']==0 or i+1==len(tr):torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();opt.zero_grad(set_to_none=True)
            met=evaluate(m,va,dev,lm);rec={'epoch':e,'train_loss':float(np.mean(running)),**{f'val_{k}':v for k,v in met.items()}};history.append(rec);print(json.dumps(rec),flush=True)
            if met['selection_score']>best:best=met['selection_score'];torch.save({'model':m.state_dict(),'config':cfg,'epoch':e,'metrics':rec},out/'best.pt')
        (out/'epoch_metrics.json').write_text(json.dumps(history,indent=2));(out/'completion.json').write_text(json.dumps({'status':'smoke' if a.smoke else 'completed','fold':fold,'epochs':epochs},indent=2))
if __name__=='__main__':main()
