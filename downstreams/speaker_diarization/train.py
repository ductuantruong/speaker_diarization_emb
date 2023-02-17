import os
import sys
sys.path.insert(0,os.getcwd())

import time
import torch
import logging
import numpy as np
from pathlib import Path
from importlib import import_module
import json
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.dataset import DiarizationDataset
from models.lightning_model import LightningModel



import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer



def collate_fn(batches):
    feat_batches = [item['feat'] for item in batches]
    label_batches = [item['label'] for item in batches]
    vector_batches = [item['spk_vector'] for item in batches]
    index_batches = [item['index_spks'] for item in batches]
    
    feat_batches = torch.stack(feat_batches)
    label_batches = torch.stack(label_batches)
    vector_batches = torch.stack(vector_batches)
    
    egs = {
        'feat': feat_batches,
        'label': label_batches,
        "spk_vector": vector_batches,
        "index_spks": index_batches,
    }
    
    return egs

def train(config): 
    # Initial
    max_epoch            = config.get('max_epoch', 100000)
    batch_size           = config.get('batch_size', 8)
    nframes              = config.get('nframes', 40)
    chunk_step           = config.get('chunk_step', 20)
    seed                 = config.get('seed', 1234)
    checkpoint_path      = config.get('checkpoint_path', '')

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   

    # Initial trainer
    # module = import_module('trainer.{}'.format(trainer_type), package=None)
    # TRAINER = getattr( module, 'Trainer')
    # trainer = TRAINER( train_config, model_config)

    # Load checkpoint if the path is given 
    if checkpoint_path != "":
        iteration = trainer.load_checkpoint( checkpoint_path)
        iteration += 1  # next iteration is iteration + 1
        
    # Load training data
    trainset = DiarizationDataset(
        config['train_config']['training_dir'], 
        chunk_size=nframes,
        chunk_step=chunk_step,
    )    
    train_loader = DataLoader(
        trainset, 
        num_workers=config['train_config']['num_workers'], 
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=trainset.collate_fn
    )
    
    # Load evaluation data
    evalset = DiarizationDataset(
        config['train_config']['eval_dir'],
        mode='test', 
        chunk_size=nframes,
        chunk_step=chunk_step,
    )    
    eval_loader = DataLoader(
        evalset, 
        num_workers=config['train_config']['num_workers'],
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=evalset.collate_fn
    )

    logger = WandbLogger(
        name=config['train_config']['exp_name'],
        offline=True,
        project='SpeakerDiarization'
    )

    model = LightningModel(config)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/{}'.format(config['train_config']['exp_name']),
        monitor='val/loss', 
        mode='min',
        verbose=1)
    
    
    trainer = Trainer(
        fast_dev_run=config['train_config']['dev'], 
        gpus=config['train_config']['gpu'], 
        max_epochs=max_epoch, 
        checkpoint_callback=True,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='min'
                ),
            model_checkpoint_callback
        ],
        logger=logger,
        resume_from_checkpoint=config['train_config']['model_checkpoint'],
        distributed_backend='ddp',
        auto_lr_find=True
        )

    # Fit model
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=eval_loader)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', model_checkpoint_callback.best_model_path)    


    """
    loss_log = dict()
    # while iteration <= max_iter:
    while epoch < max_epoch:
        trainer.model.train()
        for i, batch in enumerate(train_loader):
            
            iteration, loss_detail, lr = trainer.step(batch, iteration=iteration)

            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in loss_log.keys():
                    loss_log[key] = list()
                loss_log[key].append(val)
            
            # Save model per N iterations
            # if iteration % iters_per_checkpoint == 0:
            #     checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),iteration)
            #     trainer.save_checkpoint( checkpoint_path)

            # Show log per M iterations
            if iteration % iters_per_log == 0 and len(loss_log.keys()) > 0:
                mseg = 'Iter {}:'.format( iteration)
                for key,val in loss_log.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                mseg += '  lr: {:.6f}'.format(lr)
                logger.info(mseg)
                loss_log = dict()

            # if iteration > max_iter:
            #     break
            
        epoch += 1

        if epoch % epochs_per_eval == 0:
            eval_loss = []
            trainer.model.eval()
            for i, batch in enumerate(eval_loader):
                with torch.no_grad():
                    for key in batch.keys():
                        if key != "index_spks":
                            batch[key] = batch[key].to("cuda:1")
                    preds = trainer.model(batch)
                    targets = batch["label"]
                    bs, num_frames = targets.shape[0:2]
                    loss_batches = []
                    for idx, idx_batch in enumerate(batch["index_spks"]):
                        loss_batches.append(torch.nn.BCELoss(reduction='sum')(preds[idx, :, idx_batch], batch["label"][idx, :, idx_batch]) / num_frames)
                    loss = torch.stack(loss_batches).mean()
                    # loss = nn.BCELoss(reduction='sum')(preds, targets) / num_frames / bs
                    eval_loss.append(loss.item())
            mseg = 'Epoch {}:'.format( epoch)
            mseg += "Eval loss: {}".format(np.mean(eval_loss))
            logger.info(mseg)

        checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),epoch)
        trainer.save_checkpoint( checkpoint_path)

        if epoch > max_epoch:
            break
        

    print('Finished')
    """
        

if __name__ == "__main__":

    import argparse
    import json

    import psutil
    process = psutil.Process(os.getpid())
    process.nice(psutil.IOPRIO_CLASS_RT)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/wavlm_clustering.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    train(config)
