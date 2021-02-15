from typing import Iterable, Tuple, Dict
import argparse
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
import torch
from tqdm import tqdm
import nltk


def put_on_gpu(encoding: BatchEncoding, device: int) -> Dict[str, torch.Tensor]:
    res = {}
    for k, v in encoding.items():
        res[k] = v.cuda(device)
    return res


def batch_generator(template, pos_pairs, insert_idx, tokenizer,
                    device, batch_size, num_mask_tokens_in_template):
    sents, exp_words = [], []
    for pair in pos_pairs:
        last_words = nltk.word_tokenize(
            pair[insert_idx])[-num_mask_tokens_in_template:]
        sent = template.format(' '.join(last_words))
        sents.append(sent)

        # NB: all expected words are saved and counted
        # but not more than num_mask_token_in_batch
        new_expected_words = tokenizer.encode(
            pair[1-insert_idx], add_special_tokens=False)
        # (1) fill with mask_tokens if relation too short
        for _ in range(num_mask_tokens_in_template - len(new_expected_words)):
            new_expected_words.append(tokenizer.mask_token_id)
        # (2) cut tokens if relation too long
        new_expected_words = new_expected_words[-num_mask_tokens_in_template:]

        exp_words.extend(new_expected_words)
        if len(sents) == batch_size:
            expw_tensor = torch.LongTensor(exp_words).cuda(device)
            sents_enc = tokenizer(sents, padding=True, truncation=True,
                                  return_tensors='pt')
            sents_enc = put_on_gpu(sents_enc, device)
            mask_token_mask = sents_enc['input_ids'] == tokenizer.mask_token_id
            # mask_idx = torch.argmax(mask_token_mask.long(), dim=1)
            yield sents_enc, mask_token_mask, expw_tensor
            sents.clear()
            exp_words.clear()
    if sents:
        expw_tensor = torch.LongTensor(exp_words).cuda(device)
        sents_enc = tokenizer(sents, padding=True, truncation=True,
                              return_tensors='pt')
        sents_enc = put_on_gpu(sents_enc, device)
        mask_token_mask = sents_enc['input_ids'] == tokenizer.mask_token_id
        # mask_idx = torch.argmax(mask_token_mask.long(), dim=1)
        yield sents_enc, mask_token_mask, expw_tensor


def count_hits(masked_sentence_template: str, lm_model: AutoModelForMaskedLM, k: int,
               tokenizer: PreTrainedTokenizer,
               pos_pairs: Iterable[Tuple[str, str]], insert_idx: int,
               device: int, batch_size: int, num_mask_tokens_in_template: int) -> int:
    batches = batch_generator(
        masked_sentence_template,
        pos_pairs, insert_idx, tokenizer, device,
        batch_size, num_mask_tokens_in_template
    )
    hits = 0
    for batch in batches:
        masked_sent, mask_token_mask, expected_word = batch
        out = lm_model(**masked_sent)
        # (batch_size, seq_len, vocab_size)
        logits = out[0]
        # (num_mask_token_in_batch, vocab_size)
        mask_logits = logits[mask_token_mask]
        # (num_mask_token_in_batch, k)
        scores, indices = mask_logits.topk(k)
        hits_per_mask_token = (indices == expected_word.unsqueeze(
            1).expand_as(indices)).sum().item() / mask_logits.size(0)

        # # (batch_size, vocab_size)
        # mask_logits = torch.gather(
        #     logits, 1, mask_idx[:, None, None].expand_as(logits)
        # )[:, 0, :]
        # # (batch_size, k)
        # scores, indices = mask_logits.topk(k)

        hits += hits_per_mask_token
    return hits


def score_pattern(pattern: str, pos_pairs: Iterable[Tuple[str, str]],
                  lm_model: AutoModelForMaskedLM, tokenizer: PreTrainedTokenizer,
                  device: int, k: int = 100,
                  batch_size: int = 2, longest_mask_span: int = 1) -> int:
    hits = 0
    for num_masks in range(1, longest_mask_span+1):
        masks = ' '.join([tokenizer.mask_token] * num_masks)
        prem_masked_pattern = pattern.format(
            prem=masks, hypo='{}'
        )
        prem_hits = count_hits(prem_masked_pattern, lm_model,
                               k, tokenizer, pos_pairs, 1,
                               device, batch_size, num_masks)
        hypo_masked_pattern = pattern.format(
            hypo=masks, prem='{}')
        hypo_hits = count_hits(hypo_masked_pattern, lm_model,
                               k, tokenizer, pos_pairs, 0,
                               device, batch_size, num_masks)
        hits += prem_hits
        hits += hypo_hits

    return hits


def main(args: argparse.Namespace):
    pos_pairs = []
    with open(args.data_file) as f:
        for line in f:
            hypo, prem, cls = line.strip().split('\t')
            cls = cls == 'True'
            prem = prem.split(',')
            hypo = hypo.split(',')
            if args.ent_cls != cls:
                continue
            pos_pairs.append((prem[1], hypo[1]))

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
            k=args.k, batch_size=args.batch_size, longest_mask_span=args.longest_mask_span
        )
        pattern_score[pat] = score

    with open(args.scored_pattern_file, 'w') as fout:
        for pat in sorted(pattern_score.keys(), key=pattern_score.__getitem__, reverse=True):
            print(pattern_score[pat], pat, sep='\t', file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pattern_file')
    parser.add_argument('data_file')
    parser.add_argument('scored_pattern_file')
    parser.add_argument('--negative-class',
                        action='store_false', dest='ent_cls')
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-string', default='roberta-base')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--longest-first', action='store_true')
    parser.add_argument('--longest-mask-span', type=int, default=1)
    args = parser.parse_args()

    main(args)
