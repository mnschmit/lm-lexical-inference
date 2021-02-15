import argparse
from ..models.nli_model import NLIModel
from .utils import add_generic_args, generic_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    NLIModel.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer, model = generic_train(NLIModel, args)

    print("=== Best VAL Performance ====")
    trainer.test(test_dataloaders=model.val_dataloader())
