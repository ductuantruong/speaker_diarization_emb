import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import pandas as pd
import numpy as np
import torch_optimizer as optim

from models.models import TransformerDiarization
from itertools import permutations


def pit_loss(pred, label):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
      sigma: permutation
    """

    device = pred.device
    T = len(label)
    C = label.shape[-1]
    label_perms_indices = [
            list(p) for p in permutations(range(C))]
    P = len(label_perms_indices)
    perm_mat = torch.zeros(P, T, C, C).to(device)

    for i, p in enumerate(label_perms_indices):
        perm_mat[i, :, torch.arange(label.shape[-1]), p] = 1

    x = torch.unsqueeze(torch.unsqueeze(label, 0), -1).to(device)
    y = torch.arange(P * T * C).view(P, T, C, 1).to(device)

    broadcast_label = torch.broadcast_tensors(x, y)[0]
    allperm_label = torch.matmul(
            perm_mat, broadcast_label
            ).squeeze(-1)

    x = torch.unsqueeze(pred, 0)
    y = torch.arange(P * T).view(P, T, 1)
    broadcast_pred = torch.broadcast_tensors(x, y)[0]

    # broadcast_pred: (P, T, C)
    # allperm_label: (P, T, C)
    losses = F.binary_cross_entropy_with_logits(
               broadcast_pred,
               allperm_label,
               reduction='none')
    mean_losses = torch.mean(torch.mean(losses, dim=1), dim=1)
    min_loss = torch.min(mean_losses) * len(label)
    min_index = torch.argmin(mean_losses)
    sigma = list(permutations(range(label.shape[-1])))[min_index]

    return min_loss, allperm_label[min_index], sigma


def batch_pit_loss(ys, ts, ilens=None):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      sigmas: B-length list of permutation
    """
    if ilens is None:
        ilens = [t.shape[0] for t in ts]

    loss_w_labels_w_sigmas = [pit_loss(y[:ilen, :], t[:ilen, :])
                              for (y, t, ilen) in zip(ys, ts, ilens)]
    losses, _, sigmas = zip(*loss_w_labels_w_sigmas)
    loss = torch.sum(torch.stack(losses))
    n_frames = np.sum([ilen for ilen in ilens])
    loss = loss / n_frames

    return loss, sigmas


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        
        self.model = TransformerDiarization(**config['model_config'])
            
        self.diar_criterion = torch.nn.BCELoss()

        self.lr = config['train_config']['lr']
        self.lr_scheduler = config['train_config']['lr_scheduler']

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.model.parameters(),
                                            lr=self.lr,
                                            betas=(0.5,0.999),
                                            weight_decay=0.0)
        sch = torch.optim.lr_scheduler.StepLR(
                                optimizer=optimizer,
                                **self.lr_scheduler
                            )
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
                
            }
        }
        
    def training_step(self, batch, batch_idx):
        xs, ts, ss, ns, ilens = batch
        ys, spksvecs = self(xs)

        loss_dict = self.model.get_loss(batch, ys, spksvecs)
        
        return loss_dict
    
    def training_epoch_end(self, outputs):
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        spk_loss = torch.tensor([x['spk_loss'] for x in outputs]).mean()
        pit_loss = torch.tensor([x['pit_loss'] for x in outputs]).mean()
        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/pit_loss' , pit_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/spk_loss' , spk_loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        xs, ts, ss, ns, ilens = batch
        ys, spksvecs = self(xs)

        loss_dict = self.model.get_loss(batch, ys, spksvecs)
        
        return loss_dict

    def validation_epoch_end(self, outputs):
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        spk_loss = torch.tensor([x['spk_loss'] for x in outputs]).mean()
        pit_loss = torch.tensor([x['pit_loss'] for x in outputs]).mean()
        self.log('val/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/pit_loss' , pit_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/spk_loss' , spk_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass
