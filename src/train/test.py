import argparse
import pytorch_lightning as pl
from ..models.nli_model import NLIModel
from ..models.multnat_model import MultNatModel
from .utils import add_dataset_specific_args, load_custom_data

str2model = {
    "NLI": NLIModel,
    "MultNat": MultNatModel
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=str2model.keys())
    parser.add_argument('checkpoint')
    parser.add_argument('--classification_threshold', type=float, default=None)
    parser.add_argument('--gpus', type=int, nargs='+', default=[])
    add_dataset_specific_args(parser)
    args = parser.parse_args()

    cls = str2model[args.model]
    model = cls.load_from_checkpoint(args.checkpoint)

    if args.classification_threshold is not None:
        model.set_classification_threshold(args.classification_threshold)

    dataloader = load_custom_data(args, args.model, model)

    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.test(model, test_dataloaders=dataloader)
