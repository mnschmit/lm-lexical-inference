import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import precision, recall,\
    precision_recall_curve
from pytorch_lightning.metrics.functional import f1
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import os
import sys
import logging
from torch.utils.data import DataLoader
from .utils import compute_auc
from ..data.sherliic import SherliicSentences
from ..data.levy_holt import LevyHoltSentences


logger = logging.getLogger(__name__)


class NLIModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.save_hyperparameters(hparams)

        try:
            self.classification_threshold = self.hparams.classification_threshold
        except AttributeError:
            self.classification_threshold = 0.5

        try:
            self.minimum_precision = self.hparams.minimum_precision
        except AttributeError:
            self.minimum_precision = 0.5

        self.step_count = 0
        self.tfmr_ckpts = {}
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name
            if self.hparams.config_name else self.hparams.model_name_or_path,
            num_labels=2,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name
            if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )

        self.score_outfile = None

    def set_classification_threshold(self, thr):
        self.classification_threshold = thr

    def set_minimum_precision(self, min_prec):
        self.minimum_precision = min_prec

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def forward(self, encoded, labels):
        outputs = self.model(**encoded, labels=labels)
        loss, logits = outputs[:2]
        return loss, logits

    def training_step(self, batch, batch_idx):
        encoded, labels = batch
        loss, logits = self.forward(encoded, labels)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        encoded, labels = batch
        loss, logits = self.forward(encoded, labels)

        scores = F.softmax(logits.detach(), dim=1)

        return {'val_loss': loss.detach(), 'scores': scores, 'truth': labels}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()

        scores = torch.cat([x['scores'] for x in outputs], 0)
        truth = torch.cat([x['truth'] for x in outputs])

        pred = (scores[:, 1] > self.classification_threshold).long()

        f1_neg, f1_pos = f1(pred, truth, 2, average=None)
        prec_neg, prec_pos = precision(
            pred, truth, num_classes=2, class_reduction=None)
        rec_neg, rec_pos = recall(
            pred, truth, num_classes=2, class_reduction=None)

        prec, rec, _ = precision_recall_curve(scores[:, 1], truth)
        area_under_pr_rec_curve = compute_auc(
            prec, rec,
            filter_threshold=self.minimum_precision
        )

        metrics = {
            'F1': f1_pos, 'val_loss': val_loss_mean,
            'Precision': prec_pos, 'Recall': rec_pos,
            'AUC': area_under_pr_rec_curve
        }

        return {'val_loss': val_loss_mean, 'AUC': area_under_pr_rec_curve,
                'F1': f1_pos, 'log': metrics}

    def set_score_outfile(self, fname):
        self.score_outfile = fname
        with open(fname, 'w') as fout:
            print('score', 'label', sep='\t', file=fout)

    def test_step(self, batch, batch_idx):
        res = self.validation_step(batch, batch_idx)
        if self.score_outfile is not None:
            with open(self.score_outfile, 'a') as fout:
                for s, t in zip(res['scores'][:, 1], res['truth']):
                    print(s.item(), t.item(), sep='\t', file=fout)
        return res

    def test_epoch_end(self, outputs):
        eval_results = self.validation_epoch_end(outputs)
        return eval_results['log']

    def setup(self, stage):
        if self.hparams.levy_holt:
            if self.hparams.augment:
                self.print(
                    "WARNING: The Levy/Holt dataset does not support data augmentation.",
                    file=sys.stderr
                )

            self.train_dataset = LevyHoltSentences(
                os.path.join(self.hparams.data_dir, 'levy_holt', 'train.txt')
            )
            self.val_dataset = LevyHoltSentences(
                os.path.join(self.hparams.data_dir, 'levy_holt', 'dev.txt')
            )
            self.test_dataset = LevyHoltSentences(
                os.path.join(self.hparams.data_dir, 'levy_holt', 'test.txt')
            )
        else:
            self.train_dataset = SherliicSentences(
                os.path.join(self.hparams.data_dir, 'sherliic', 'train.csv'),
                with_examples=True,
                augment=self.hparams.augment
            )
            self.val_dataset = SherliicSentences(
                os.path.join(self.hparams.data_dir, 'sherliic', 'dev.csv'),
                with_examples=True,
                augment=False
            )
            self.test_dataset = SherliicSentences(
                os.path.join(self.hparams.data_dir, 'sherliic', 'test.csv'),
                with_examples=True,
                augment=False
            )

        train_batch_size = self.hparams.train_batch_size
        self.total_steps = (
            (len(self.train_dataset) //
             (train_batch_size * max(1, len(self.hparams.gpus))))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )

    def collate(self, samples):
        prems, hypos, labels = map(list, zip(*samples))
        encoded = self.tokenizer(
            prems, hypos,
            padding=True, return_tensors='pt'
        )
        label_tensor = torch.LongTensor(labels)
        return encoded, label_tensor

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate,
            pin_memory=True
        )

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str,
            help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--data_dir", default=os.path.join(os.getcwd(), 'data'), type=str
        )
        parser.add_argument("--learning_rate", default=5e-5,
                            type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0,
                            type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8,
                            type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0,
                            type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=2,
                            type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs",
                            dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)

        parser.add_argument("--augment", action='store_true')

        parser.add_argument("--minimum_precision", type=float, default=0.5)
        parser.add_argument("--classification_threshold",
                            type=float, default=0.5)

        parser.add_argument("--levy_holt", action='store_true')

        return parser
