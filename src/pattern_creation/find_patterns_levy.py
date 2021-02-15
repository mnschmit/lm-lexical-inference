from typing import Iterable, List, Tuple
import argparse
import re
from multiprocessing import Pool
import os
from functools import partial
from tqdm import tqdm


def process_sentence_file(
        pos_regex: re._pattern_type,
        neg_regex: re._pattern_type,
        files: Tuple[str, str, str],
        prem_markers=('*_', '_*'), hypo_markers=('>>', '<<'),
        prem_first=True
) -> Tuple[int, int]:

    if prem_first:
        modified_template = '{}' + \
            prem_markers[0]+'{}'+prem_markers[1]+'{}' + \
            hypo_markers[0]+'{}'+hypo_markers[1]+'{}'
    else:
        modified_template = '{}' + \
            hypo_markers[0]+'{}'+hypo_markers[1]+'{}' + \
            prem_markers[0]+'{}'+prem_markers[1]+'{}'

    file_in, file_out_pos, file_out_neg = files
    pos_matches, neg_matches = [], []
    with open(file_in) as f:
        for sent in f:
            for regex, match_storage in zip((pos_regex, neg_regex), (pos_matches, neg_matches)):
                match = regex.match(sent)
                if match:
                    modsent_buffer = []
                    for g in match.groups():
                        if g is not None:
                            modsent_buffer.append(g)
                    modified_sent = modified_template.format(
                        *modsent_buffer)
                    match_storage.append(modified_sent)

    if pos_matches:
        with open(file_out_pos, 'w') as fout:
            for sent in pos_matches:
                print(sent, file=fout)
    if neg_matches:
        with open(file_out_neg, 'w') as fout:
            for sent in neg_matches:
                print(sent, file=fout)

    return len(pos_matches), len(neg_matches)


def create_pattern_from_word_pair(prem: str, hypo: str) -> str:
    # NB: word boundaries at beginning and end of relations
    return r'(.*)\b({})\b(.*\s+\S+\s+.*)\b({})\b(.*)'.format(prem, hypo)


def create_pattern_from_dataset(
        rows: Iterable[Tuple[str, str]], prem_first: bool
) -> str:
    sentence_patterns = []
    for prem, hypo in rows:
        if prem_first:
            sentence_pattern = create_pattern_from_word_pair(
                prem, hypo
            )
        else:
            sentence_pattern = create_pattern_from_word_pair(
                hypo, prem
            )

        sentence_patterns.append(sentence_pattern)

    return '|'.join(sentence_patterns)


def extract_instances_from_dataset(
        file_name: str
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    pos_inst, neg_inst = [], []
    with open(file_name) as f:
        for line in f:
            hypo, prem, cls = line.strip().split('\t')
            prem = prem.split(',')
            hypo = hypo.split(',')
            if cls == 'True':
                pos_inst.append((prem[1], hypo[1]))
            else:
                neg_inst.append((prem[1], hypo[1]))
    return pos_inst, neg_inst


def find_patterns(args, prem_first):
    pos_inst, neg_inst = extract_instances_from_dataset(args.dataset_file)
    pos_pattern = create_pattern_from_dataset(pos_inst, prem_first)
    neg_pattern = create_pattern_from_dataset(neg_inst, prem_first)
    pos_regex = re.compile(pos_pattern)
    neg_regex = re.compile(neg_pattern)

    sentence_file_tuples = []
    for in_fn in args.sentences_in:
        path_elements = in_fn.split(os.path.sep)
        base_fn = path_elements[-2] + '_' + path_elements[-1]
        out_fn1 = os.path.join(
            args.out_dir, base_fn + '-pos-' + ('prem_first' if prem_first else 'hypo_first'))
        out_fn2 = os.path.join(
            args.out_dir, base_fn + '-neg-' + ('prem_first' if prem_first else 'hypo_first'))
        sentence_file_tuples.append((in_fn, out_fn1, out_fn2))
    sentence_file_tuples = tqdm(sentence_file_tuples)

    global_pcount = 0
    global_ncount = 0
    if args.num_threads > 1:
        with Pool(processes=args.num_threads) as pool:
            for pcount, ncount in pool.imap_unordered(
                    partial(process_sentence_file, pos_regex,
                            neg_regex, prem_first=prem_first),
                    sentence_file_tuples
            ):
                global_pcount += pcount
                global_ncount += ncount
    else:
        for sftup in sentence_file_tuples:
            pcount, ncount = process_sentence_file(
                pos_regex, neg_regex, sftup, prem_first=prem_first)
            global_pcount += pcount
            global_ncount += ncount

    return global_pcount, global_ncount


def main(args: argparse.Namespace):
    pcount1, ncount1 = find_patterns(args, True)
    pcount2, ncount2 = find_patterns(args, False)

    print(
        "Found {} positive and {} negative matches.".format(
            pcount1+pcount2, ncount1+ncount2
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    parser.add_argument('sentences_in', nargs='+')
    parser.add_argument('out_dir')
    parser.add_argument('num_threads', type=int)
    args = parser.parse_args()

    main(args)
