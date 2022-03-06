from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import gc

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_fn(dataloader,model,optimizer,scaler,device,scheduler,epoch):
    '''Perform model training'''
    model.train()
    loss_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i,batch in tk0:
        optimizer.zero_grad()

        for k in batch:
            batch[k] = batch[k].to(device)
        batch_size = len(batch['input'])
        
        with torch.cuda.amp.autocast():
            out = model(batch)

        loss, target, conf, cls, embeddings = out['loss'], out['target'], out['preds_conf'], out['preds_cls'], out['embeddings']
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg,Epoch=epoch,LR=optimizer.param_groups[0]['lr'])
        del batch, loss, target, conf, cls, embeddings 

        gc.collect()
        torch.cuda.empty_cache()
        
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss_score.avg)
    elif scheduler is not None:
        scheduler.step()
        
    return loss_score.avg