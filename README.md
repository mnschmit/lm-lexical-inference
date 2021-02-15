# Language Models for Lexical Inference in Context
This repository contains the code and instructions to reproduce the results from the EACL 2021 long paper "Language Models for Lexical Inference in Context".


# Data
## Levy/Holt and SherLIiC (including data splits)
The data set splits as used in the experiments can be downloaded from [here](cistern.cis.lmu.de/lm-lexical-inference).
After extraction, the data should be organized into a directory structure like this (the 'patterns' directory is already part of this repository):

```
data
├── levy_holt
│   ├── dev.txt
│   ├── test.txt -> ../../raw/levy_holt/test_s2.txt
│   └── train.txt
├── patterns
│   ├── levyholt-neg.txt
│   ├── levyholt-pos.txt
│   ├── sherliic-neg.txt
│   ├── sherliic-pos-curated-args.txt
│   ├── sherliic-pos-curated.txt
│   └── sherliic-pos.txt
└── sherliic
    ├── dev.csv
    ├── test.csv -> ../../raw/sherliic/test.csv
    └── train.csv
```

## Patterns from Wikipedia
The pattern files (in data/patterns) are licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)


# Code

## Training
Training a model is done with the following two commands:
1. `src/train/nli_model.py` for the NLI model
2. `src/train/multnat_model.py` for the pattern-based models

Here is an example for training a RoBERTa-base AUTPAT5 model that uses antipatterns on the Levy/Holt data:
```
CUDA_VISIBLE_DEVICES=<gpu_num> python3 -m src.train.multnat_model --gpus 0 --model_name_or_path roberta-base --experiment_name roberta-base_autpat5-wanti_levy-holt --num_train_epochs 5 --checkpoint_dir checkpoints/ --gradient_accumulation_steps 2 --learning_rate 3.82e-5 --train_batch_size 10 --weight_decay 4.02e-5 --pattern_file data/patterns/levyholt-pos.txt --antipattern_file data/patterns/levyholt-neg.txt --best_k_patterns 5 --use_antipatterns --levy_holt --num_workers 1
```
Explanations:
- `<gpu_num>`: Which GPU to use, e.g., 0
- `--levy_holt` is a flag that has to be set if the data to be trained on is from the Levy/Holt data set. Otherwise the data format from SherLIiC is assumed.
- `--num_workers 1` is only needed for AUTPAT models. For MANPAT models, it can be omitted or set to a higher number.

For MANPAT models, the optional arguments `--pattern_file`, `--antipattern_file`, and `--best_k_patterns` simply have to be removed, i.e., without given pattern/antipattern files the handcrafted ones are the default.

## Testing
You can use a model checkpoint to evaluate on its respective test split or a different data set with the following command:
```
CUDA_VISIBLE_DEVICES=<gpu_num> python3 -m src.train.test <model_type> checkpoint.ckpt --gpus 0 --classification_threshold <custom_threshold> --dataset <path_to_dataset> --levy_holt --pattern_file data/patterns/pattern_files/wiki06-pos-scored-v2.txt --antipattern_file data/patterns/pattern_files/wiki06-neg-scored-v2.txt --best_k_patterns 5
```

Explanations:
- `<model_type>`: One of either `NLI` or `MultNat`
- `<custom_threshold>`: Some float number for classification decisions, e.g., 0.5 (see section below)
- `<path_to_dataset>`: e.g., `data/levy_holt/test.txt`
- `--levy_holt` is a flag that has to be set if the data to be evaluated on is from the Levy/Holt data set. Otherwise the data format from SherLIiC is assumed.
- The optional arguments `--pattern_file`, `--antipattern_file`, and `--best_k_patterns` are only relevant, when an AUTPAT model is to be evaluated, and should otherwise be omitted.
- If the optional `--dataset` argument is not given, the default is to evaluate on the respective test set.

## Write Scores /  Tune Classification Threshold
When you want to tune a classification threshold yourself, you first have to write out the raw scores of a trained model like this:
```
CUDA_VISIBLE_DEVICES=<gpu_num> python3 -m src.train.write_scores MultNat checkpoint.ckpt score_file.tsv --gpus 0
```
This `score_file.tsv` corresponds, by default, to the respective dev set and can be used to find the best threshold like this:
```
python3 src/train/tune_threshold.py score_file.tsv
```

If you want to produce the score file for different data than the dev set, you can use the same arguments to specify a data set as explained above for the test script.

## Error Analysis
Score files, whose creation is explained in the previous section, can also be used for qualitative error analysis.
1. `scr/analysis/errors.py` lists false positives and false negatives together with their scores and line numbers.
2. `src/analysis/classifications_from_scores.py` allows to quickly compare the predictions of multiple models on a single example (like Table 6), given the line number of said example.


# Additional Results on Transfer Learning
The original paper does not investigate how well a model trained on one data set performs on the other. Here we present more results on that.
Note that in this setting we assume that the target data set is not available at all, i.e., we do not use it at all - neither for finding patterns in AUTPAT nor for tuning the classification threshold. We use the standard thresholds, i.e., 0.5 for NLI and 0.0 for the pattern-based methods.


## Trained on SherLIiC, evaluated on Levy/Holt test

| RoBERTa-base |  AUC |  Prec |  Rec |   F1 |
| ------------ | ----:| -----:| ----:| ----:|
| NLI          | 38.4 | 52.7  | 57.1 | **54.8** |
| MANPAT ΦΨ    | **46.1** | **64.0**  | 45.4 | 53.1 |
| MANPAT Φ     | 32.4 | 32.4  | **94.5** | 48.2 |
| AUTPAT-5 ΦΨ  | 18.7 | 40.5  | 35.0 | 37.6 |
| AUTPAT-5 Φ   | 21.3 | 28.3  | 62.3 | 38.9 |


| RoBERTa-large |  AUC | Prec |  Rec |   F1 |
| ------------- | ----:| ----:| ----:| ----:|
| NLI           | 37.8 | 31.0 | 96.4 | 46.9 |
| MANPAT ΦΨ     | **70.4** | 39.6 | 95.3 | **56.0** |
| MANPAT Φ      | 38.9 | 25.6 | **98.3** | 40.6 |
| AUTPAT-5 ΦΨ   | 33.6 | **61.6** | 36.0 | 45.5 |
| AUTPAT-5 Φ    |  9.3 | 30.7 | 76.6 | 43.8 |


## Trained on Levy/Holt, evaluated on SherLIiC test

| RoBERTa-base |  AUC | Prec |  Rec |   F1 |
| ------------ | ----:| ----:| ----:| ----:|
| NLI          | 63.3 | 62.8 | **68.4** | **65.5** |
| MANPAT ΦΨ    | **69.1** | **80.5** | 42.1 | 55.3 |
| MANPAT Φ     | 68.4 | 80.1 | 24.2 | 37.2 |
| AUTPAT-5 ΦΨ  | 60.3 | 71.5 | 54.5 | 61.9 |
| AUTPAT-5 Φ   | 58.9 | 68.6 | 55.7 | 61.5 |


| RoBERTa-large |  AUC | Prec |  Rec |   F1 |
| ------------- | ----:| ----:| ----:| ----:|
| NLI           | 65.6 | 73.8 | 53.0 | 61.7 |
| MANPAT ΦΨ     | 69.6 | 84.7 | 35.7 | 50.3 |
| MANPAT Φ      | **72.2** | **89.3** | 30.3 | 45.2 |
| AUTPAT-5 ΦΨ   | 62.1 | 68.1 | **57.3** | **62.3** |
| AUTPAT-5 Φ    | 63.8 | 75.8 | 44.2 | 55.8 |

