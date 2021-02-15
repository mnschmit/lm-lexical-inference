from typing import Optional, List, TypeVar, Iterable
import re

PREM_KEY = 'premise'
HYPO_KEY = 'hypothesis'
LABEL_KEY = 'label'
SENT_KEY = 'sentence'
ANTI_KEY = 'neg_sentence'
MASKED_SENT_KEY = 'masked_sentence'
MASKED_ANTI_KEY = 'masked_neg_sentence'


PATTERNS = [
    "{pal} {prem} {par}, which means that {hal} {hypo} {har}.",
    "It is not the case that {hal} {hypo} {har}, let alone that {pal} {prem} {par}.",
    "{hal} {hypo} {har} because {pal} {prem} {par}.",
    "{pal} {prem} {par} because {hal} {hypo} {har}.",
    "{hal} {hypo} {har}, which means that {pal} {prem} {par}."
]
NEGATION_NECESSARY = [
    (False, False),
    (False, False),
    (False, False),
    (True, True),
    (True, True)
]
ANTIPATTERNS = [
    "It is not sure that {hal} {hypo} {har} just because {pal} {prem} {par}.",
    "{pal} {prem} {par}. This does not mean that {hal} {hypo} {har}.",
    "The fact that {pal} {prem} {par} does not necessarily mean that {hal} {hypo} {har}.",
    "Even if {pal} {prem} {par}, {hal} maybe {hypo} {har}.",
    "Just because {pal} {prem} {par}, it might still not be true that {hal} {hypo} {har}."
]
ANTI_NEGATION_NECESSARY = [
    (False, False),
    (False, False),
    (False, False),
    (False, True),
    (False, False)
]


def choose_examples(examples_A, examples_B, is_reversed: bool):
    if is_reversed:
        return examples_B[0], examples_A[0]
    else:
        return examples_A[0], examples_B[0]


def negate(verb_phrase: str) -> str:
    tokens = re.split(r'\s+', verb_phrase)
    if tokens[0] in ['is', 'are', 'were', 'was']:
        new_tokens = tokens[:1] + ['not'] + tokens[1:]
    else:
        if tokens[0].endswith('s'):
            new_tokens = ['does', 'not', tokens[0][:-1]] + tokens[1:]
        else:
            new_tokens = ['do', 'not', tokens[0][:-1]] + tokens[1:]
    return ' '.join(new_tokens)


def mask_equivalent(self, string: str, mask_token, tokenizer, add_space=True) -> str:
    longer_string = mask_token
    if add_space:
        longer_string = longer_string + ' '
    longer_string = longer_string + string.strip()
    num_tokens = len(
        tokenizer.encode(longer_string, add_special_tokens=False)
    ) - 1
    return " ".join([mask_token] * num_tokens)


def load_patterns(pattern_file: str, best_k_patterns: Optional[int]) -> List[str]:
    patterns = []
    with open(pattern_file) as f:
        for line in f:
            score, pattern = line.strip().split('\t')
            patterns.append(pattern)
            if best_k_patterns is not None and len(patterns) == best_k_patterns:
                break
    return patterns


T = TypeVar('T')


def chunks(lst: List[T], n: int) -> Iterable[List[T]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
