import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import pandas as pd
import torch_optimizer as optim

from models.models import TransformerDiarization


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

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.model.parameters(),
                                            lr=self.lr,
                                            betas=(0.5,0.999),
                                            weight_decay=0.0)
        sch = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer,
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
        input = batch
        for key in batch.keys():
            if key != "index_spks":
                input[key] = input[key].to(self.device)

        preds = self(input)
        bs, tframe = input["label"].shape[0:2]

        loss_batches = []
        for idx, idx_batch in enumerate(input["index_spks"]):
            loss_batches.append(self.diar_criterion(reduction='sum')(preds[idx, :, idx_batch], input["label"][idx, :, idx_batch]) / tframe) 
        loss = torch.stack(loss_batches).mean()
        
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        for key in batch.keys():
            if key != "index_spks":
                batch[key] = batch[key]
        preds = self(batch)
        targets = batch["label"]
        bs, num_frames = targets.shape[0:2]
        loss_batches = []
        for idx, idx_batch in enumerate(batch["index_spks"]):
            loss_batches.append(self.diar_criterion(reduction='sum')(preds[idx, :, idx_batch], batch["label"][idx, :, idx_batch]) / num_frames)
        loss = torch.stack(loss_batches).mean()
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass
