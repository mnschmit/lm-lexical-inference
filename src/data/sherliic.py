from typing import Sequence, List, Dict, Union, Tuple, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import csv
import sys
from tqdm import tqdm
import logging
from overrides import overrides
from itertools import product
import re
from .common import PREM_KEY, HYPO_KEY, LABEL_KEY, SENT_KEY, ANTI_KEY,\
    MASKED_SENT_KEY, MASKED_ANTI_KEY, PATTERNS, NEGATION_NECESSARY, ANTIPATTERNS,\
    ANTI_NEGATION_NECESSARY, choose_examples, negate, mask_equivalent, load_patterns,\
    chunks


def unpack_row(row: Sequence[str])\
    -> Tuple[
        int, int, int, int, int,
        str, str, str, str,
        str, str, str, str,
        bool, bool,
        Tuple[str, str, str], Tuple[str, str, str],
        bool, float, float,
        float, int]:
    sample_id, prem_type, prem_rel, hypo_type, hypo_rel,\
        prem_argleft, prem_middle, prem_argright, prem_end,\
        hypo_argleft, hypo_middle, hypo_argright, hypo_end,\
        is_prem_reversed, is_hypo_reversed,\
        examples_A, examples_B, gold_label,\
        rel_score, sign_score, esr_score,\
        num_disagr = row

    return int(sample_id), int(prem_type), int(prem_rel), int(hypo_type), int(hypo_rel),\
        prem_argleft, prem_middle, prem_argright, prem_end,\
        hypo_argleft, hypo_middle, hypo_argright, hypo_end,\
        is_prem_reversed == 'True', is_hypo_reversed == 'True',\
        tuple(examples_A.split(' / ')), tuple(examples_B.split(' / ')),\
        gold_label == 'yes', float(rel_score), float(sign_score),\
        float(esr_score), int(num_disagr)


class SherliicBase(Dataset):

    def __init__(self, path_to_csv: str):
        super().__init__()
        self.fn = path_to_csv
        self.data = self.load_dataset()

    def load_dataset(self):
        logger = logging.getLogger(__name__)
        logger.info('Loading dataset from {}'.format(self.fn))
        with open(self.fn) as f:
            cr = csv.reader(f)
            next(cr)  # headers
            data = [inst for row in tqdm(cr)
                    for inst in self.create_instances(*unpack_row(row))]
        return data

    def create_instances(
            self, sample_id: int, prem_type: int, prem_rel: int, hypo_type: int, hypo_rel: int,
            prem_argleft: str, prem_middle: str, prem_argright: str, prem_end: str,
            hypo_argleft: str, hypo_middle: str, hypo_argright: str, hypo_end: str,
            is_prem_reversed: bool, is_hypo_reversed: bool,
            examples_A: Tuple[str, str, str], examples_B: Tuple[str, str, str], gold_label: bool,
            rel_score: float, sign_score: float, esr_score: float, num_disagr: int
    ) -> List[Dict[str, Union[bool, str]]]:
        raise NotImplementedError(
            "You need to implement `create_instances` in a child class derived from `SherliicBase`"
        )

    def __getitem__(self, index):
        raise NotImplementedError(
            "You need to implement `__getitem__` in a child class derived from `SherliicBase`"
        )

    def __len__(self):
        return len(self.data)


class SherliicPattern(SherliicBase):
    def __init__(self, path_to_csv: str, tokenizer: Optional[PreTrainedTokenizer] = None,
                 pattern_file: Optional[str] = None, antipattern_file: Optional[str] = None,
                 best_k_patterns: int = 100,
                 pattern_idx: int = 0, antipattern_idx: int = 0,
                 with_examples: bool = False,
                 mask_prem_not_hypo: bool = False, augment: bool = False,
                 training: bool = False, pattern_chunk_size: int = 5,
                 curated_auto: bool = False):
        self.with_examples = with_examples
        self.mask_prem = mask_prem_not_hypo
        self.training = training
        self.pattern_chunk_size = pattern_chunk_size

        if pattern_file is None:
            self.patterns = PATTERNS
            self.handcrafted = True
            self.antipatterns = ANTIPATTERNS
            self.negation_necessary = NEGATION_NECESSARY
            self.anti_negation_necessary = ANTI_NEGATION_NECESSARY
            if antipattern_file is not None:
                print(
                    "WARNING: antipattern_file is ignored because pattern_file was not specified",
                    file=sys.stderr
                )
        else:
            if best_k_patterns is not None and best_k_patterns % pattern_chunk_size != 0:
                print(
                    "WARNING: best_k_patterns should be a"
                    + " multiple of pattern_chunk_size (default: 5)",
                    file=sys.stderr
                )
            self.patterns = load_patterns(pattern_file, best_k_patterns)
            self.handcrafted = curated_auto
            assert curated_auto or (not (with_examples or augment)),\
                "automatic patterns do not make use of relation arguments; "\
                + "thus neither examples nor augmenting are supported."
            assert antipattern_file is not None,\
                "antipattern_file has to be specified when pattern_file is used."
            self.antipatterns = load_patterns(
                antipattern_file, best_k_patterns
            )
            self.negation_necessary = [
                (False, False) for _ in self.patterns
            ]
            self.anti_negation_necessary = [
                (False, False) for _ in self.antipatterns
            ]

        self.pattern_idx = pattern_idx
        self.antipattern_idx = antipattern_idx
        assert pattern_idx < len(self.patterns),\
            "You cannot choose among more than {} patterns.".format(
                len(self.patterns)
        )
        assert antipattern_idx < len(self.antipatterns),\
            "You cannot choose among more than {} antipatterns.".format(
            len(self.antipatterns)
        )

        # TODO: implement data augmentation with different variables
        assert not augment or with_examples,\
            "data augmentation without examples is not implemented"
        self.augment = augment

        self.tokenizer = tokenizer
        if self.tokenizer:
            self.mask_token = self.tokenizer.special_tokens_map['mask_token']
        super().__init__(path_to_csv)

    def create_sent_from_pattern(self, pattern: str, premise, hypothesis,
                                 is_prem_reversed, is_hypo_reversed,
                                 examples_A, examples_B,
                                 do_negation: Tuple[bool, bool], no_mask: bool = False) -> str:
        prem_argleft, prem_middle, prem_argright, prem_end = premise
        hypo_argleft, hypo_middle, hypo_argright, hypo_end = hypothesis

        if not no_mask:
            if self.mask_prem:
                prem_middle = mask_equivalent(
                    prem_middle, self.mask_token, self.tokenizer)
                if prem_end.strip():
                    prem_end = mask_equivalent(
                        prem_end, self.mask_token, self.tokenizer, add_space=False)
            else:
                hypo_middle = mask_equivalent(
                    hypo_middle, self.mask_token, self.tokenizer)
                if hypo_end.strip():
                    hypo_end = mask_equivalent(
                        hypo_end, self.mask_token, self.tokenizer, add_space=False)

        if self.with_examples:
            prem_argleft, prem_argright = choose_examples(
                examples_A, examples_B, is_prem_reversed)
            hypo_argleft, hypo_argright = choose_examples(
                examples_A, examples_B, is_hypo_reversed)

        if do_negation[0]:
            prem_middle = negate(prem_middle)
        if do_negation[1]:
            hypo_middle = negate(hypo_middle)

        if self.handcrafted:
            inserted = pattern.format(
                pal=prem_argleft, prem=prem_middle, par=prem_argright+prem_end,
                hal=hypo_argleft, hypo=hypo_middle, har=hypo_argright+hypo_end
            )
        else:
            try:
                inserted = pattern.format(
                    prem=prem_middle + ' ' + prem_end,
                    hypo=hypo_middle + ' ' + hypo_end
                )
            except KeyError as e:
                print(pattern)
                raise e

        return re.sub(r'\s+', ' ', inserted)

    def create_single_instance(self, premise: Tuple[str, str, str, str],
                               hypothesis: Tuple[str, str, str, str],
                               is_prem_reversed: bool, is_hypo_reversed: bool,
                               examples_A: Tuple[str, str, str],
                               examples_B: Tuple[str, str, str], gold_label: bool
                               ) -> Dict[str, Union[bool, str, List[str]]]:
        inst = {
            LABEL_KEY: gold_label
        }

        if self.pattern_idx < 0:
            inst[SENT_KEY] = []
            if self.tokenizer:
                inst[MASKED_SENT_KEY] = []

            for pat, do_neg in zip(self.patterns, self.negation_necessary):
                inst[SENT_KEY].append(
                    self.create_sent_from_pattern(
                        pat,
                        premise, hypothesis,
                        is_prem_reversed, is_hypo_reversed,
                        examples_A, examples_B,
                        do_neg,
                        no_mask=True
                    )
                )
                if self.tokenizer:
                    inst[MASKED_SENT_KEY].append(
                        self.create_sent_from_pattern(
                            pat, premise, hypothesis,
                            is_prem_reversed, is_hypo_reversed,
                            examples_A, examples_B,
                            do_neg, no_mask=False
                        )
                    )
        else:
            inst[SENT_KEY] = self.create_sent_from_pattern(
                self.patterns[self.pattern_idx],
                premise, hypothesis,
                is_prem_reversed, is_hypo_reversed, examples_A, examples_B,
                self.negation_necessary[self.pattern_idx],
                no_mask=True
            )

            if self.tokenizer:
                inst[MASKED_SENT_KEY] = self.create_sent_from_pattern(
                    self.patterns[self.pattern_idx],
                    premise, hypothesis,
                    is_prem_reversed, is_hypo_reversed, examples_A, examples_B,
                    self.negation_necessary[self.pattern_idx],
                    no_mask=False
                )

        if self.antipattern_idx < 0:
            inst[ANTI_KEY] = []
            if self.tokenizer:
                inst[MASKED_ANTI_KEY] = []

            for pat, do_neg in zip(self.antipatterns, self.anti_negation_necessary):
                inst[ANTI_KEY].append(
                    self.create_sent_from_pattern(
                        pat, premise, hypothesis,
                        is_prem_reversed, is_hypo_reversed,
                        examples_A, examples_B,
                        do_neg, no_mask=True
                    )
                )
                if self.tokenizer:
                    inst[MASKED_ANTI_KEY].append(
                        self.create_sent_from_pattern(
                            pat, premise, hypothesis,
                            is_prem_reversed, is_hypo_reversed,
                            examples_A, examples_B,
                            do_neg, no_mask=False
                        )
                    )
        else:
            inst[ANTI_KEY] = self.create_sent_from_pattern(
                self.antipatterns[self.antipattern_idx],
                premise, hypothesis,
                is_prem_reversed, is_hypo_reversed, examples_A, examples_B,
                self.anti_negation_necessary[self.antipattern_idx],
                no_mask=True
            )

            if self.tokenizer:
                inst[MASKED_ANTI_KEY] = self.create_sent_from_pattern(
                    self.antipatterns[self.antipattern_idx],
                    premise, hypothesis,
                    is_prem_reversed, is_hypo_reversed, examples_A, examples_B,
                    self.anti_negation_necessary[self.antipattern_idx],
                    no_mask=False
                )

        return inst

    @overrides
    def create_instances(
            self, sample_id: int, prem_type: int, prem_rel: int, hypo_type: int, hypo_rel: int,
            prem_argleft: str, prem_middle: str, prem_argright: str, prem_end: str,
            hypo_argleft: str, hypo_middle: str, hypo_argright: str, hypo_end: str,
            is_prem_reversed: bool, is_hypo_reversed: bool,
            examples_A: Tuple[str, str, str], examples_B: Tuple[str, str, str], gold_label: bool,
            rel_score: float, sign_score: float, esr_score: float, num_disagr: int
    ) -> List[Dict[str, Union[bool, str, List[str]]]]:

        instances = []

        if self.augment:
            for chosen_A, chosen_B in product(examples_A, examples_B):
                if chosen_A == chosen_B:
                    continue
                instances.append(
                    self.create_single_instance(
                        (prem_argleft, prem_middle, prem_argright, prem_end),
                        (hypo_argleft, hypo_middle, hypo_argright, hypo_end),
                        is_prem_reversed, is_hypo_reversed,
                        (chosen_A,), (chosen_B,),
                        gold_label
                    )
                )
        else:
            inst = self.create_single_instance(
                (prem_argleft, prem_middle, prem_argright, prem_end),
                (hypo_argleft, hypo_middle, hypo_argright, hypo_end),
                is_prem_reversed, is_hypo_reversed,
                examples_A, examples_B,
                gold_label
            )
            if self.training and self.pattern_idx < 0:
                lists = self.unpack_instance(inst)
                label = lists[-1]
                chunked = [chunks(x, self.pattern_chunk_size)
                           for x in lists[:-1]]
                for chunk in zip(*chunked):
                    smaller_inst = {
                        SENT_KEY: chunk[0],
                        ANTI_KEY: chunk[1],
                        LABEL_KEY: label
                    }

                    if self.tokenizer:
                        smaller_inst[MASKED_SENT_KEY] = chunk[2]
                        smaller_inst[MASKED_ANTI_KEY] = chunk[3]

                    instances.append(smaller_inst)
            else:
                instances.append(inst)

        return instances

    def unpack_instance(
            self, inst: Dict[str, Union[str, bool, List[str]]]
    ) -> List[Union[str, bool, List[str]]]:
        if self.tokenizer:
            return [
                inst[SENT_KEY], inst[ANTI_KEY],
                inst[MASKED_SENT_KEY], inst[MASKED_ANTI_KEY],
                inst[LABEL_KEY]
            ]
        else:
            return [inst[SENT_KEY], inst[ANTI_KEY], inst[LABEL_KEY]]

    @overrides
    def __getitem__(self, index):
        inst = self.data[index]
        return self.unpack_instance(inst)


class SherliicSentences(SherliicBase):

    def __init__(self, path_to_csv, with_examples=False, augment=False):
        self.with_examples = with_examples
        self.augment = augment
        super().__init__(path_to_csv)

    def construct_from_parts(self, *parts):
        joined = " ".join([part.strip() for part in parts]).strip()
        joined = joined.replace(" 's", "'s")
        return joined + '.'

    def insert_examples(
            self, chosen_A, chosen_B, is_prem_reversed, is_hypo_reversed,
            prem_argleft, prem_middle, prem_argright, prem_end,
            hypo_argleft, hypo_middle, hypo_argright, hypo_end
    ):
        if is_prem_reversed:
            prem_argleft = chosen_B
            prem_argright = chosen_A
        else:
            prem_argleft = chosen_A
            prem_argright = chosen_B
        if is_hypo_reversed:
            hypo_argleft = chosen_B
            hypo_argright = chosen_A
        else:
            hypo_argleft = chosen_A
            hypo_argright = chosen_B

        prem = self.construct_from_parts(
            prem_argleft, prem_middle, prem_argright, prem_end
        )
        hypo = self.construct_from_parts(
            hypo_argleft, hypo_middle, hypo_argright, hypo_end
        )
        return prem, hypo

    def change_var(self, var, placeholder):
        return placeholder[:-2] + var + ']'

    @overrides
    def create_instances(
            self, sample_id: int, prem_type: int, prem_rel: int, hypo_type: int, hypo_rel: int,
            prem_argleft: str, prem_middle: str, prem_argright: str, prem_end: str,
            hypo_argleft: str, hypo_middle: str, hypo_argright: str, hypo_end: str,
            is_prem_reversed: bool, is_hypo_reversed: bool,
            examples_A: Tuple[str, str, str], examples_B: Tuple[str, str, str], gold_label: bool,
            rel_score: float, sign_score: float, esr_score: float, num_disagr: int
    ) -> List[Dict[str, Union[bool, str]]]:

        prems = []
        hypos = []
        if self.with_examples:
            if self.augment:
                for chosen_A, chosen_B in product(examples_A, examples_B):
                    if chosen_A == chosen_B:
                        continue
                    prem, hypo = self.insert_examples(
                        chosen_A, chosen_B, is_prem_reversed, is_hypo_reversed,
                        prem_argleft, prem_middle, prem_argright, prem_end,
                        hypo_argleft, hypo_middle, hypo_argright, hypo_end
                    )
                    prems.append(prem)
                    hypos.append(hypo)
            else:
                prem, hypo = self.insert_examples(
                    examples_A[0], examples_B[0], is_prem_reversed, is_hypo_reversed,
                    prem_argleft, prem_middle, prem_argright, prem_end,
                    hypo_argleft, hypo_middle, hypo_argright, hypo_end
                )
                prems.append(prem)
                hypos.append(hypo)
        else:
            if self.augment:
                for var1, var2 in [('A', 'B'), ('B', 'A'), ('X', 'Y'), ('Y', 'X')]:
                    if is_prem_reversed:
                        prem_argleft = self.change_var(var2, prem_argleft)
                        prem_argright = self.change_var(var1, prem_argright)
                    else:
                        prem_argleft = self.change_var(var1, prem_argleft)
                        prem_argright = self.change_var(var2, prem_argright)
                    if is_hypo_reversed:
                        hypo_argleft = self.change_var(var2, hypo_argleft)
                        hypo_argright = self.change_var(var1, hypo_argright)
                    else:
                        hypo_argleft = self.change_var(var1, hypo_argleft)
                        hypo_argright = self.change_var(var2, hypo_argright)

                    prems.append(self.construct_from_parts(prem_argleft, prem_middle,
                                                           prem_argright, prem_end))
                    hypos.append(self.construct_from_parts(hypo_argleft, hypo_middle,
                                                           hypo_argright, hypo_end))
            else:
                prems.append(self.construct_from_parts(prem_argleft, prem_middle,
                                                       prem_argright, prem_end))
                hypos.append(self.construct_from_parts(hypo_argleft, hypo_middle,
                                                       hypo_argright, hypo_end))

        instances = []
        for p, h in zip(prems, hypos):
            instances.append({
                PREM_KEY: p,
                HYPO_KEY: h,
                LABEL_KEY: gold_label
            })
        return instances

    @overrides
    def __getitem__(self, index):
        inst = self.data[index]
        return inst[PREM_KEY], inst[HYPO_KEY], inst[LABEL_KEY]
