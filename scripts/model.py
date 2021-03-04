from dataset import DataSet
from config import Config

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup

from utils import ramp_scheduler


# * Encoder/Features extractor: bert/roberta/distilbert multilingual
# * Decoder/Classifier : linear layer (bert_out_features, n_classes)
# * loss_fn : CrossEntropyLoss
# * Optimize metrics : accuracy
# * Optimizer : Adam, AdamW, SGD
# * learning rate : (3e-5...1e-1)
# * lr scheduler : linear with warmup, ReduceLROnPlateau
# * pretrained : Always true


class Model(pl.LightningModule):
    def __init__(
            self,
            class_w=None,
            base_model: str = Config.base_model,
            freeze: bool = False):
        super(Model, self).__init__()

        self.save_hyperparameters()
        # load pretrained model from torchvision or anywheare else
        try:
            self.encoder = AutoModel.from_pretrained(Config.base_model)
            # print(self.encoder)
        except Exception as e:
            print(f'{e}')

        if "distil" in self.hparams.base_model:
            try:
                # get num output features from features extractor
                self.num_ftrs = self.encoder.transformer.layer[-1].ffn.lin2.out_features
            except:
                pass

        elif "roberta" in self.hparams.base_model:
            try:
                self.num_ftrs = self.encoder.pooler.dense.out_features
            except Exception as e:
                print(f'{e}')

        else:
            try:
                self.num_ftrs = self.encoder.pooler.dense.out_features
            except Exception as e:
                print(f'{e}')

        if self.hparams.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=.35)
        self.decoder = nn.Linear(
            in_features=self.num_ftrs,
            out_features=Config.n_classes
        )

        #########
        # methods
        #########

    def forward(self, ids, mask, targets=None):
        # extract features
        _, out = self.encoder(ids, mask)
        # apply dropout
        out = self.dropout(out)
        # apply classifier
        out = self.decoder(out)

        return out

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.001
             },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
             },
        ]

        opt = th.optim.SGD(
            optimizer_parameters,
            lr=self.hparams.lr
        )

        scheduler = th.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=ramp_scheduler,
            verbose=True
        )

        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        ids, mask, targets = batch['ids'], batch['mask'], batch['target']
        # make predictions
        predictions = self(ids, mask)

        # compute metrics
        # loss
        loss = self.get_loss(preds=predictions, targets=targets)

        # accuracy
        acc = self.get_acc(preds=predictions, targets=targets)

        # logging stuff
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": predictions,
                'targets': targets}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        ids, mask, targets = batch['ids'], batch['mask'], batch['target']
        # make predictions
        predictions = self(ids, mask)

        # compute metrics
        # loss
        loss = self.get_loss(preds=predictions, targets=targets)

        # accuracy
        acc = self.get_acc(preds=predictions, targets=targets)

        # logging stuff
        self.log('val_acc', acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": predictions,
                'targets': targets}

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def predict(self, dataloader, batch_size=1):
        if batch_size == 1:
            try:
                preds = self(dataloader.unsqueeze(0))
            except:
                preds = self(dataloader)
        else:
            preds = self(dataloader)

        return preds.detach().cpu().numpy().flatten()

    def get_loss(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return nn.CrossEntropyLoss(weight=class_w)(input=preds, target=targets)

    def get_acc(self, preds, targets):
        preds = preds.cpu()
        targets = targets.cpu()
        return accuracy(pred=preds, target=targets)


if __name__ == "__main__":
    print('[INFO] Building model')
    try:
        model = Model(
            class_w=None
        )

        print(model)
    except Exception as e:
        print(f'[ERROR] {e}')
