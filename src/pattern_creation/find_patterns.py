from typing import Iterable, List, Tuple, Dict
import argparse
import re
import csv
from multiprocessing import Pool
import os
from functools import partial
from tqdm import tqdm
from verb_forms import is_verb_form, is_non_verb, verb_tenses, verb_infinitive, is_verb_inf,\
    verb_past_participle, verb_present, verb_present_participle, verb_past


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


def generate_forms(verb_inf: str) -> List[str]:
    forms = [verb_inf]
    pres3sg = verb_present(verb_inf, person='3')
    if pres3sg:
        forms.append(pres3sg)

    be = generate_forms_of_be()
    gerund = verb_present_participle(verb_inf)
    if gerund:
        forms.append(be + ' ' + gerund)

    past = verb_past(verb_inf, person='3')
    if past:
        forms.append(past)
    return forms


def generate_forms_of_be() -> List[str]:
    return '(?:' + "|".join(
        sorted([form for form in verb_tenses['be']
                if form], key=len, reverse=True)
    ) + ')'


def create_pattern_from_relation_path(relpath: str, is_reversed: bool) -> str:
    lemmas = [
        word
        for i, word in enumerate(relpath.split('___'))
        if i % 2 == 1
    ]
    if is_reversed:
        lemmas.reverse()

    pred_is_verb = is_verb_inf(lemmas[0]) and not is_non_verb(lemmas[0])

    if (relpath.startswith('nsubjpass') or relpath.endswith('nsubjpass')
            or relpath.endswith('nsubjpass^-')) and pred_is_verb:
        be = generate_forms_of_be()
        pred = verb_past_participle(lemmas[0])
        pred = be + ' ' + pred
    else:
        if pred_is_verb:
            pred = '(?:' + "|".join(
                sorted(generate_forms(lemmas[0]), key=len, reverse=True)
            ) + ')'
        else:
            pred = generate_forms_of_be()
            lemmas.insert(0, 'be')

    poss_involved = relpath.startswith('poss') or relpath.endswith(
        'poss') or relpath.endswith('poss^-')

    if len(lemmas) == 1:
        return pred
    elif len(lemmas) == 2:
        if poss_involved:
            return pred + r'(?:\s*\S+\s*){0,5}' + r"'s\s*" + lemmas[1]
        else:
            return pred + ' ' + lemmas[1]
    else:
        if poss_involved:
            return (pred + ' ' + ' '.join(lemmas[1:-1])
                    + r'(?:\s*\S+\s*){0,5}' + r"'s\s*" + lemmas[-1])
        else:
            return pred + ' ' + ' '.join(lemmas[1:])


def create_pattern_from_relation(rel_middle: str, rel_end: str) -> str:
    rel_end = rel_end.strip()
    rel_middle = rel_middle.strip()

    if rel_middle == 'is':
        return rel_end

    # is preming -> preming
    # is supporter -> supporter
    if rel_middle.startswith('is '):
        rel_middle = rel_middle[3:]
        # if rel_middle.endswith('ing') and len(rel_middle) > 5:
        #     rel_middle = rel_middle[:-3]

    pred, *rest = rel_middle.split()

    # prem -> prem|preming|prems|premed
    if is_verb_form(pred) and not is_non_verb(pred):
        inf = verb_infinitive(pred)
        rel_middle = '(?:' + "|".join(
            sorted([form for form in verb_tenses[inf]
                    if form], key=len, reverse=True)
        ) + ')'
        if rest:
            rel_middle += ' ' + ' '.join(rest)

    if not rel_end:
        return rel_middle
    else:
        return rel_middle + r'(?:\s*\S+\s*){0,5}' + rel_end


def create_pattern_from_dataset(
        rows: Iterable[Tuple[str, bool, str, bool]], prem_first: bool
) -> str:
    sentence_patterns = []
    for prem_path, is_prem_reversed, hypo_path, is_hypo_reversed in rows:
        prem_pattern = create_pattern_from_relation_path(
            prem_path, is_prem_reversed)
        hypo_pattern = create_pattern_from_relation_path(
            hypo_path, is_hypo_reversed)
        if prem_first:
            sentence_pattern = create_pattern_from_word_pair(
                prem_pattern, hypo_pattern
            )
        else:
            sentence_pattern = create_pattern_from_word_pair(
                hypo_pattern, prem_pattern
            )

        sentence_patterns.append(sentence_pattern)

    return '|'.join(sentence_patterns)


def extract_instances_from_dataset(
        file_name: str, relation_index: Dict[int, str]) -> Tuple[
            List[Tuple[str, bool, str, bool]],
            List[Tuple[str, bool, str, bool]]
]:
    pos_inst, neg_inst = [], []
    with open(file_name) as f:
        reader = csv.reader(f)
        next(reader)  # headers
        for row in reader:
            # prem = tuple(row[5:9])
            # hypo = tuple(row[9:13])
            prem_path = relation_index[row[2]]
            hypo_path = relation_index[row[4]]
            is_prem_reversed = row[13] == 'True'
            is_hypo_reversed = row[14] == 'True'
            cls = row[17] == 'yes'
            if cls:
                pos_inst.append((prem_path, is_prem_reversed,
                                 hypo_path, is_hypo_reversed))
            else:
                neg_inst.append((prem_path, is_prem_reversed,
                                 hypo_path, is_hypo_reversed))
    return pos_inst, neg_inst


def find_patterns(args, rel_idx, prem_first):
    pos_inst, neg_inst = extract_instances_from_dataset(
        args.dataset_file, rel_idx
    )
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
    rel_idx = {}
    with open(args.relation_index) as f:
        for line in f:
            idx, relpath = line.strip().split('\t')
            rel_idx[idx] = relpath

    pcount1, ncount1 = find_patterns(args, rel_idx, True)
    pcount2, ncount2 = find_patterns(args, rel_idx, False)

    print(
        "Found {} positive and {} negative matches.".format(
            pcount1+pcount2, ncount1+ncount2
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    parser.add_argument('relation_index')
    parser.add_argument('sentences_in', nargs='+')
    parser.add_argument('out_dir')
    parser.add_argument('num_threads', type=int)
    args = parser.parse_args()

    main(args)
