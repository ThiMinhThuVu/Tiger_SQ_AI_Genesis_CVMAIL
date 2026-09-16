#!/usr/bin/env python3
"""ConvNeXt-V2 Tiny/Base + Mask2Former multitask 5-fold baseline."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.convnextv2_mask2former import build_model
from tiger_models.data import LabelMap,TigerMultitaskDataset,records_for_fold,visibility_pos_weight,STATIONS
from tiger_models.metrics import segmentation_metrics,visibility_metrics,selection_score
from tiger_models.monai_mask2former_loss import build_monai_dice_focal_loss,mask2former_monai_loss

def collate(items):
    masks=[]; labels=[]
    for item in items:
        target=item['fine'].long(); classes=torch.unique(target)
        masks.append(torch.stack([(target==c).float() for c in classes])); labels.append(classes)
    return {'image':torch.stack([x['image'] for x in items]),'fine':torch.stack([x['fine'] for x in items]),
            'visibility':torch.stack([x['visibility'] for x in items]),'mask_labels':masks,'class_labels':labels,
            'case_id':[x['case_id'] for x in items],'name':[x['name'] for x in items]}

def semantic_logits(output):
    classes=output.class_queries_logits.softmax(-1)[...,:-1]; masks=output.masks_queries_logits.sigmoid()
    return torch.einsum('bqc,bqhw->bchw',classes,masks)

@torch.no_grad()
def evaluate(model,loader,device,lm):
    model.eval(); fp=[];ft=[];cases=[];vt=[];vp=[]
    for b in loader:
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            out,vis=model(b['image'].to(device))
        logits=F.interpolate(semantic_logits(out).float(),size=b['fine'].shape[-2:],mode='bilinear',align_corners=False)
        fp.extend(logits.argmax(1).cpu().numpy());ft.extend(b['fine'].numpy());cases.extend(b['case_id']);vt.append(b['visibility'].numpy());vp.append(vis.float().sigmoid().cpu().numpy())
    fine=segmentation_metrics(fp,ft,cases,lm.fine_weights,lm.fine_names);cp=[lm.fine_to_coarse[x] for x in fp];ct=[lm.fine_to_coarse[x] for x in ft];coarse=segmentation_metrics(cp,ct,cases,lm.coarse_weights,lm.coarse_names);v=visibility_metrics(np.concatenate(vt),np.concatenate(vp),STATIONS);score=selection_score(fine['dice'],fine['nhd'],coarse['dice'],coarse['nhd'],v['macro_f1'],v['macro_auroc'])
    return {'fine_dice':fine['dice'],'fine_nhd':fine['nhd'],'coarse_dice':coarse['dice'],'coarse_nhd':coarse['nhd'],'visibility_f1':v['macro_f1'],'visibility_auroc':v['macro_auroc'],'selection_score':score}

def model_from_cfg(cfg,pretrained):
    return build_model(cfg['model_name'],pretrained,cfg['num_labels'],cfg['decoder_layers'],cfg['num_queries'])

def run_fold(cfg,fold,a,device,lm):
    random.seed(cfg['seed']+fold);np.random.seed(cfg['seed']+fold);torch.manual_seed(cfg['seed']+fold)
    train_records=records_for_fold(a.data_root,fold,'train');trds=TigerMultitaskDataset(train_records,lm,cfg['height'],cfg['width'],True);vlds=TigerMultitaskDataset(records_for_fold(a.data_root,fold,'validation'),lm,cfg['height'],cfg['width'],False);kw={'num_workers':cfg['num_workers'],'pin_memory':device.type=='cuda','collate_fn':collate};tr=DataLoader(trds,batch_size=cfg['batch_size'],shuffle=True,**kw);va=DataLoader(vlds,batch_size=1,shuffle=False,**kw)
    model=model_from_cfg(cfg,cfg['pretrained']).to(device);enc=list(model.backbone.parameters());enc_ids={id(p) for p in enc};heads=[p for p in model.parameters() if id(p) not in enc_ids];opt=torch.optim.AdamW([{'params':enc,'lr':cfg['encoder_learning_rate']},{'params':heads,'lr':cfg['learning_rate']}],weight_decay=cfg['weight_decay']);pos=torch.from_numpy(visibility_pos_weight(train_records)).to(device);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');outdir=a.output_root/cfg['experiment_id']/f'fold_{fold}';outdir.mkdir(parents=True,exist_ok=True);best=-1.;history=[];epochs=1 if a.smoke else cfg['epochs'];monai_weight=float(cfg.get('monai_aux_loss_weight',0.));monai_criterion=build_monai_dice_focal_loss() if monai_weight>0 else None
    for epoch in range(epochs):
        model.train();opt.zero_grad(set_to_none=True);running=[];running_native=[];running_monai=[]
        for step,b in enumerate(tr):
            image=b['image'].to(device); target=b['visibility'].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                seg,vis=model(image,mask_labels=[x.to(device) for x in b['mask_labels']],class_labels=[x.to(device) for x in b['class_labels']]);vloss=F.binary_cross_entropy_with_logits(vis,target,pos_weight=pos)
            mloss=mask2former_monai_loss(seg,b['fine'].to(device),monai_criterion) if monai_criterion is not None else seg.loss.new_zeros(());native=seg.loss+vloss;total=native+monai_weight*mloss
            scaler.scale(total/cfg['gradient_accumulation']).backward();running.append(float(total.detach().cpu()));running_native.append(float(native.detach().cpu()));running_monai.append(float(mloss.detach().cpu()))
            if (step+1)%cfg['gradient_accumulation']==0 or step+1==len(tr):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        metrics=evaluate(model,va,device,lm);row={'epoch':epoch,'train_loss':float(np.mean(running)),'train_native_loss':float(np.mean(running_native)),'train_monai_aux_loss':float(np.mean(running_monai)),**{f'val_{k}':v for k,v in metrics.items()}};history.append(row);print(json.dumps(row),flush=True)
        if metrics['selection_score'] is not None and metrics['selection_score']>best:best=metrics['selection_score'];torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch,'metrics':row},outdir/'best.pt')
    (outdir/'epoch_metrics.json').write_text(json.dumps(history,indent=2));(outdir/'completion.json').write_text(json.dumps({'status':'smoke' if a.smoke else 'completed','fold':fold,'epochs':epochs},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--run-all-folds',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-cpu',action='store_true');p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output-root',type=Path,default=ROOT/'artifacts/new_baselines');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
    if not torch.cuda.is_available() and not a.allow_cpu:raise RuntimeError('CUDA unavailable; use --allow-cpu only for smoke')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');lm=LabelMap.load(a.data_root/'labelmap.csv');folds=range(5) if a.run_all_folds else [a.fold]
    for fold in folds:run_fold(cfg,fold,a,device,lm)
if __name__=='__main__':main()
