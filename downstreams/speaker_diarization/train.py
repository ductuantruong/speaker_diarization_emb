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
from functools import partial

from torch.utils.data import DataLoader
from downstreams.speaker_diarization.utils.dataset import DiarizationDataset
from models.lightning_model import LightningModel



import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

def collate_fn_ns(batch, n_speakers, spkidx_tbl):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    valid_chunk_indices1 = [i for i in range(len(ts))
                            if ts[i].shape[1] == n_speakers]
    valid_chunk_indices2 = []

    # n_speakers (rec-data) > n_speakers (model)
    invalid_chunk_indices1 = [i for i in range(len(ts))
                              if ts[i].shape[1] > n_speakers]

    ts = list(ts)
    ss = list(ss)
    for i in invalid_chunk_indices1:
        s = np.sum(ts[i], axis=0)
        cs = ts[i].shape[0]
        if len(s[s > 0.5]) <= n_speakers:
            # n_speakers (chunk-data) <= n_speakers (model)
            # update valid_chunk_indices2
            valid_chunk_indices2.append(i)
            idx_arr = np.where(s > 0.5)[0]
            ts[i] = ts[i][:, idx_arr]
            ss[i] = ss[i][idx_arr]
            if len(s[s > 0.5]) < n_speakers:
                # n_speakers (chunk-data) < n_speakers (model)
                # update ts[i] and ss[i]
                n_speakers_real = len(s[s > 0.5])
                zeros_ts = np.zeros((cs, n_speakers), dtype=np.float32)
                zeros_ts[:, :-(n_speakers-n_speakers_real)] = ts[i]
                ts[i] = zeros_ts
                mones_ss = -1 * np.ones((n_speakers,), dtype=np.int64)
                mones_ss[:-(n_speakers-n_speakers_real)] = ss[i]
                ss[i] = mones_ss
            else:
                # n_speakers (chunk-data) == n_speakers (model)
                pass
        else:
            # n_speakers (chunk-data) > n_speakers (model)
            pass

    # valid_chunk_indices: chunk indices using for training
    valid_chunk_indices = sorted(valid_chunk_indices1 + valid_chunk_indices2)

    ilens = np.array(ilens)
    ilens = ilens[valid_chunk_indices]
    ns = np.array(ns)[valid_chunk_indices]
    ss = np.array([ss[i] for i in range(len(ss))
                  if ts[i].shape[1] == n_speakers])
    xs = [xs[i] for i in range(len(xs)) if ts[i].shape[1] == n_speakers]
    ts = [ts[i] for i in range(len(ts)) if ts[i].shape[1] == n_speakers]
    xs = np.array([np.pad(x, [(0, np.max(ilens) - len(x))],
                          'constant', constant_values=(-1,)) for x in xs])
    ts = np.array([np.pad(t, [(0, np.max(ilens) - len(t)), (0, 0)],
                          'constant', constant_values=(+1,)) for t in ts])

    if spkidx_tbl is not None:
        # Update global speaker ID
        all_n_speakers = np.max(spkidx_tbl) + 1
        bs = len(ns)
        ns = np.array([
                np.arange(
                    all_n_speakers,
                    dtype=np.int64
                    ).reshape(all_n_speakers, 1)] * bs)
        ss = np.array([spkidx_tbl[ss[i]] for i in range(len(ss))])

    return (xs, ts, ss, ns, ilens)



def collate_fn(batch):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    ilens = np.array(ilens)

    # import time
    # time.sleep(1999)
    xs = np.array([np.pad(
        x, [(0, np.max(ilens) - len(x))],
        'constant', constant_values=(-1,)
        ) for x in xs])
    ts = np.array([np.pad(
        t, [(0, np.max(ilens) - len(t)), (0, 0)],
        'constant', constant_values=(+1,)
        ) for t in ts])
    ss = np.array(ss)
    ns = np.array(ns)

    return (xs, ts, ss, ns, ilens)


def train(config): 
    # Initial
    max_epoch            = config.get('max_epoch', 100000)
    batch_size           = config.get('batch_size', 4)
    seed                 = config.get('seed', 1234)
    checkpoint_path      = config.get('checkpoint_path', None)

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   

    
    # Load training data
    trainset = DiarizationDataset(
        mode= 'train',
        data_dir=config['train_config']['training_dir'], 
        chunk_size=750,
        frame_shift=320,
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
        mode= 'train',
        data_dir=config['train_config']['eval_dir'],
        chunk_size=750,
        frame_shift=320,
    )    
    eval_loader = DataLoader(
        evalset, 
        num_workers=config['train_config']['num_workers'],
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=evalset.collate_fn_infer
    )

    """
    logger = WandbLogger(
        name=config['train_config']['exp_name'],
        offline=True,
        project='SpeakerDiarization'
    )
    """

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
        num_sanity_val_steps=0,
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
        #logger=logger,
        resume_from_checkpoint=checkpoint_path,
        distributed_backend='dp',
        auto_lr_find=True
        )
    
    if checkpoint_path is not None:
        iteration = trainer.load_checkpoint(checkpoint_path)
        iteration += 1  # next iteration is iteration + 1
    
    # Fit model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', model_checkpoint_callback.best_model_path)    



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
