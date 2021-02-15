from typing import Iterable, Tuple, Dict
import argparse
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
import torch
import csv
from tqdm import tqdm


def put_on_gpu(encoding: BatchEncoding, device: int) -> Dict[str, torch.Tensor]:
    res = {}
    for k, v in encoding.items():
        res[k] = v.cuda(device)
    return res


def batch_generator(template, pos_pairs, insert_idx, tokenizer, device, batch_size):
    sents, exp_words = [], []
    for pair in pos_pairs:
        sent = template.format(pair[insert_idx])
        sents.append(sent)
        expected_word: int = tokenizer.encode(
            pair[1-insert_idx], add_special_tokens=False)[0]
        exp_words.append(expected_word)
        if len(exp_words) == batch_size:
            expw_tensor = torch.LongTensor(exp_words).cuda(device)
            sents_enc = tokenizer(sents, padding=True, truncation=True,
                                  return_tensors='pt')
            sents_enc = put_on_gpu(sents_enc, device)
            mask_token_mask = sents_enc['input_ids'] == tokenizer.mask_token_id
            mask_idx = torch.argmax(mask_token_mask.long(), dim=1)
            yield sents_enc, mask_idx, expw_tensor
            sents.clear()
            exp_words.clear()
    if sents:
        expw_tensor = torch.LongTensor(exp_words).cuda(device)
        sents_enc = tokenizer(sents, padding=True, truncation=True,
                              return_tensors='pt')
        sents_enc = put_on_gpu(sents_enc, device)
        mask_token_mask = sents_enc['input_ids'] == tokenizer.mask_token_id
        mask_idx = torch.argmax(mask_token_mask.long(), dim=1)
        yield sents_enc, mask_idx, expw_tensor


def count_hits(masked_sentence_template: str, lm_model: AutoModelForMaskedLM, k: int,
               tokenizer: PreTrainedTokenizer,
               pos_pairs: Iterable[Tuple[str, str]], insert_idx: int,
               device: int, batch_size: int) -> int:
    batches = batch_generator(
        masked_sentence_template,
        pos_pairs, insert_idx, tokenizer, device,
        batch_size
    )
    hits = 0
    for batch in batches:
        masked_sent, mask_idx, expected_word = batch
        out = lm_model(**masked_sent)
        # (batch_size, seq_len, vocab_size)
        logits = out[0]
        # (batch_size, vocab_size)
        mask_logits = torch.gather(
            logits, 1, mask_idx[:, None, None].expand_as(logits)
        )[:, 0, :]
        # (batch_size, k)
        scores, indices = mask_logits.topk(k)
        hits += (indices == expected_word.unsqueeze(1).expand_as(indices)).sum().item()
    return hits


def score_pattern(pattern: str, pos_pairs: Iterable[Tuple[str, str]],
                  lm_model: AutoModelForMaskedLM, tokenizer: PreTrainedTokenizer,
                  device: int, k: int = 100, batch_size: int = 2) -> int:
    prem_masked_pattern = pattern.format(prem=tokenizer.mask_token, hypo='{}')
    prem_hits = count_hits(prem_masked_pattern, lm_model,
                           k, tokenizer, pos_pairs, 1, device, batch_size)
    hypo_masked_pattern = pattern.format(hypo=tokenizer.mask_token, prem='{}')
    hypo_hits = count_hits(hypo_masked_pattern, lm_model,
                           k, tokenizer, pos_pairs, 0, device, batch_size)
    return prem_hits + hypo_hits


def main(args: argparse.Namespace):
    rel_idx = {}
    with open(args.relation_index) as f:
        for line in f:
            idx, rel = line.strip().split('\t')
            rel_idx[idx] = rel

    pos_pairs = []
    with open(args.sherliic_file) as f:
        r = csv.reader(f)
        next(r)  # headers
        for row in r:
            cls = row[17] == 'yes'
            if args.ent_cls != cls:
                continue
            prem_path = rel_idx[row[2]]
            hypo_path = rel_idx[row[4]]
            prem_idx = -2 if row[13] == 'True' else 1
            hypo_idx = -2 if row[14] == 'True' else 1
            prem = prem_path.split('___')[prem_idx]
            hypo = hypo_path.split('___')[hypo_idx]
            pos_pairs.append((prem, hypo))

    lm_model = AutoModelForMaskedLM.from_pretrained(args.model_string)
    tokenizer = AutoTokenizer.from_pretrained(args.model_string)

    lm_model.cuda(args.gpu)

    patterns = []
    with open(args.pattern_file) as f:
        for pat in f:
            patterns.append(pat.strip())
    if args.longest_first:
        patterns.sort(key=len, reverse=True)

    pattern_score = {}
    for pat in tqdm(patterns):
        score = score_pattern(
            pat, pos_pairs, lm_model, tokenizer, args.gpu,
            k=args.k, batch_size=args.batch_size
        )
        pattern_score[pat] = score

    with open(args.scored_pattern_file, 'w') as fout:
        for pat in sorted(pattern_score.keys(), key=pattern_score.__getitem__, reverse=True):
            print(pattern_score[pat], pat, sep='\t', file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pattern_file')
    parser.add_argument('sherliic_file')
    parser.add_argument('relation_index')
    parser.add_argument('scored_pattern_file')
    parser.add_argument('--negative-class',
                        action='store_false', dest='ent_cls')
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--model-string', default='roberta-base')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--longest-first', action='store_true')
    args = parser.parse_args()

    main(args)
