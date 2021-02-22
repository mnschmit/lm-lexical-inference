from typing import List, Dict, Union
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1_score
from pytorch_lightning.metrics.functional.classification import precision, recall,\
    precision_recall_curve
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BatchEncoding
)
import os
import sys
import logging
from torch.utils.data import DataLoader
from .utils import compute_auc
from ..data.sherliic import SherliicPattern
from ..data.levy_holt import LevyHoltPattern


logger = logging.getLogger(__name__)


class MultNatModel(pl.LightningModule):
    def __init__(self, hparams: Union[argparse.Namespace,
                                      Dict[str, Union[int, bool, float, str]]]):
        super().__init__()

        self.save_hyperparameters(hparams)

        try:
            self.classification_threshold = self.hparams.classification_threshold
        except AttributeError:
            self.classification_threshold = 0.0

        try:
            self.minimum_precision = self.hparams.minimum_precision
        except AttributeError:
            self.minimum_precision = 0.5

        try:
            self.curated_auto = self.hparams.curated_auto
        except AttributeError:
            self.curated_auto = False

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

    def forward(self, encs, anti_encs, labels):
        # length of these lists will be num_sents
        losses, list_of_logits = [], []
        for enc in encs:
            out = self.model(**enc, labels=labels)
            loss, logits = out[:2]
            losses.append(loss)
            list_of_logits.append(logits.detach().unsqueeze(1))
        if losses:
            loss1 = sum(losses) / len(losses)
            # (batch_size, num_sents, num_classes)
            logits = torch.cat(list_of_logits, 1)
        else:
            loss1 = 0.0
            logits = None

        if self.hparams.use_antipatterns:
            losses, list_of_logits = [], []
            for anti_enc in anti_encs:
                out = self.model(**anti_enc, labels=1 - labels)
                loss, anti_logits = out[:2]
                losses.append(loss)
                list_of_logits.append(anti_logits.detach().unsqueeze(1))
            if losses:
                loss2 = sum(losses) / len(losses)
                anti_logits = torch.cat(list_of_logits, 1)
            else:
                loss2 = 0.0
                anti_logits = None
        else:
            loss2 = 0.0
            anti_logits = None

        loss = loss1 + loss2

        # logits.view(bs, num_sents, -1)
        return loss, logits, anti_logits

    def training_step(self, batch, batch_idx):
        enc, anti_enc, labels = batch
        loss, logits, anti_logits = self.forward(enc, anti_enc, labels)

        # return loss
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        enc, anti_enc, labels = batch
        loss, logits, anti_logits = self.forward(enc, anti_enc, labels)

        # (batch_size, num_sents, num_classes)
        probs = F.softmax(logits, dim=2)
        # (batch_size,)
        max_prob, _ = torch.max(probs[:, :, 1], dim=1)

        if anti_logits is None:
            min_prob, _ = torch.max(probs[:, :, 0], dim=1)
            scores = max_prob - min_prob  # (max_prob > min_prob).long()
        else:
            anti_probs = F.softmax(anti_logits, dim=2)
            anti_max_prob, _ = torch.max(anti_probs[:, :, 1], dim=1)
            scores = max_prob - anti_max_prob

        return {'val_loss': loss.detach(), 'scores': scores, 'truth': labels}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()

        scores = torch.cat([x['scores'] for x in outputs], 0)
        truth = torch.cat([x['truth'] for x in outputs], 0)

        pred = (scores > self.classification_threshold).long()

        f1_neg, f1_pos = f1_score(pred, truth, num_classes=2, reduction='none')
        prec_neg, prec_pos = precision(
            pred, truth, num_classes=2, reduction='none')
        rec_neg, rec_pos = recall(
            pred, truth, num_classes=2, reduction='none')

        prec, rec, _ = precision_recall_curve(scores, truth)
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
                for s, t in zip(res['scores'], res['truth']):
                    print(s.item(), t.item(), sep='\t', file=fout)
        return res

    def test_epoch_end(self, outputs):
        eval_results = self.validation_epoch_end(outputs)
        print(eval_results['log'])
        return eval_results['log']

    def setup(self, stage):
        if self.hparams.levy_holt:
            if self.hparams.augment:
                self.print(
                    "WARNING: The Levy/Holt dataset does not support data augmentation.",
                    file=sys.stderr
                )

            self.train_dataset = LevyHoltPattern(
                os.path.join(self.hparams.data_dir, 'levy_holt', 'train.txt'),
                pattern_file=self.hparams.pattern_file,
                antipattern_file=self.hparams.antipattern_file,
                best_k_patterns=self.hparams.best_k_patterns,
                training=True,
                curated_auto=self.curated_auto
            )
            self.val_dataset = LevyHoltPattern(
                os.path.join(self.hparams.data_dir, 'levy_holt', 'dev.txt'),
                pattern_file=self.hparams.pattern_file,
                antipattern_file=self.hparams.antipattern_file,
                best_k_patterns=self.hparams.best_k_patterns,
                training=False,
                curated_auto=self.curated_auto
            )
            self.test_dataset = LevyHoltPattern(
                os.path.join(self.hparams.data_dir, 'levy_holt', 'test.txt'),
                pattern_file=self.hparams.pattern_file,
                antipattern_file=self.hparams.antipattern_file,
                best_k_patterns=self.hparams.best_k_patterns,
                training=False,
                curated_auto=self.curated_auto
            )
        else:
            self.train_dataset = SherliicPattern(
                os.path.join(self.hparams.data_dir, 'sherliic', 'train.csv'),
                pattern_idx=-1,
                antipattern_idx=-1,
                with_examples=True,
                augment=self.hparams.augment,
                pattern_file=self.hparams.pattern_file,
                antipattern_file=self.hparams.antipattern_file,
                best_k_patterns=self.hparams.best_k_patterns,
                training=True,
                curated_auto=self.curated_auto
            )
            self.val_dataset = SherliicPattern(
                os.path.join(self.hparams.data_dir, 'sherliic', 'dev.csv'),
                pattern_idx=-1,
                antipattern_idx=-1,
                with_examples=True,
                pattern_file=self.hparams.pattern_file,
                antipattern_file=self.hparams.antipattern_file,
                best_k_patterns=self.hparams.best_k_patterns,
                training=False,
                curated_auto=self.curated_auto
            )
            self.test_dataset = SherliicPattern(
                os.path.join(self.hparams.data_dir, 'sherliic', 'test.csv'),
                pattern_idx=-1,
                antipattern_idx=-1,
                with_examples=True,
                pattern_file=self.hparams.pattern_file,
                antipattern_file=self.hparams.antipattern_file,
                best_k_patterns=self.hparams.best_k_patterns,
                training=False,
                curated_auto=self.curated_auto
            )

        train_batch_size = self.hparams.train_batch_size
        self.total_steps = (
            (len(self.train_dataset) //
             (train_batch_size * max(1, len(self.hparams.gpus))))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )

    def encode_batch_of_sentences(self, sents: List[List[str]]) -> List[BatchEncoding]:
        restructured_sents = []
        for i in range(len(sents[0])):
            new_list = []
            for slist in sents:
                new_list.append(slist[i])
            restructured_sents.append(new_list)

        sents_enc = []
        for batch_sized_slist in restructured_sents:
            sents_enc.append(
                self.tokenizer(
                    batch_sized_slist, truncation=True,
                    padding=True, return_tensors='pt'
                )
            )
        return sents_enc

    def collate(self, samples):
        sents, antisents, labels = map(list, zip(*samples))

        sents_enc = self.encode_batch_of_sentences(sents)
        antisents_enc = self.encode_batch_of_sentences(antisents)

        label_tensor = torch.LongTensor(labels)
        return sents_enc, antisents_enc, label_tensor

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
        parser.add_argument("--train_batch_size", default=20, type=int)
        parser.add_argument("--eval_batch_size", default=10, type=int)

        parser.add_argument("--classification_threshold",
                            default=0.0, type=float)
        parser.add_argument("--minimum_precision", default=0.5, type=float)

        parser.add_argument("--use_antipatterns", action='store_true')
        parser.add_argument("--augment", action='store_true')
        parser.add_argument("--pattern_file", default=None)
        parser.add_argument("--antipattern_file", default=None)
        parser.add_argument("--best_k_patterns", type=int, default=100)
        parser.add_argument("--curated_auto", action='store_true')

        parser.add_argument("--levy_holt", action='store_true')

        return parser
