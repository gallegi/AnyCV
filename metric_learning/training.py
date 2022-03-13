from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import gc

from commons.training import AverageMeter

def train_fn(dataloader,model,optimizer,scaler,device,scheduler,epoch):
    """Performs training one epoch
    
    Args:
        dataloader: a loader that fetch minibatch each time
        model: a training model
        optimizer: an optimizer that update the model's weight using gradients
        scaler: a GradScaler object that scale the gradients
        device: a device that the model performs forward and backward passes on
        scheduler: a learning rate controller
        epoch: the current training epoch

    Returns:
        The average loss value of the current epoch
    """
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

def get_all_embeddings(dataloader, model, device='cuda:0'):
    """Get embedding array on the whole dataset

    Args:
        dataloader: a loader that fetch a minibatch each time
        model: a model that map images to embedding vectors
        device: a device that the model performs

    Returns:
        Embeddings matrix (N x emb_size) where N is the total number of images
        and emb_size is the embeddings size
    """
    model.eval()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    all_embeddings = []
    for i,batch in tk0:
        
        for k in batch:
            batch[k] = batch[k].to(device)
        batch_size = len(batch['input'])
        
        with torch.no_grad():
            out = model(batch)
        loss, target, conf, cls, embeddings = out['loss'], out['target'], out['preds_conf'], out['preds_cls'], out['embeddings']
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings)

def get_all_embeddings_and_predictions(dataloader, model, device='cuda:0'):
     """Get embedding array on the whole dataset

    Args:
        dataloader: a loader that fetch a minibatch each time
        model: a model that map images to embedding vectors
        device: a device that the model performs

    Returns:
        A dictionary that contains those keys:
            embeddings: Embeddings matrix (N x emb_size) where N is the total number of images
                        and emb_size is the embeddings size
            pred_classes: predicted class for all images (N)
            pred_conf: corresponding confidence scores of the class prediction
    """
    model.eval()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    all_embeddings = []
    all_pred_classes = []
    all_pred_confs = []
    for i,batch in tk0:
        
        for k in batch:
            batch[k] = batch[k].to(device)
        batch_size = len(batch['input'])
        
        with torch.no_grad():
            out = model(batch)
        loss, target, conf, cls, embeddings = out['loss'], out['target'], out['preds_conf'], out['preds_cls'], out['embeddings']
        all_embeddings.append(embeddings.cpu().numpy())
        all_pred_classes.append(cls.cpu().numpy())
        all_pred_confs.append(conf.cpu().numpy())

    out_dict = {'embeddings': np.concatenate(all_embeddings),
                'pred_class':np.concatenate(all_pred_classes), 
                'pred_conf': np.concatenate(all_pred_confs)}
    return out_dict