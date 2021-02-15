import argparse
import re
from tqdm import tqdm

PREM_MARKERS = ('*_', '_*')
HYPO_MARKERS = ('>>', '<<')


def clean_instantiated_pattern(
        pattern: str,
        regex: re._pattern_type
) -> str:
    match = regex.match(pattern)
    assert match is not None

    if len(match.groups()) != 10:
        print("Expected 10 captured groups but did not find them!")
        print(pattern)
        print(match.groups())
        exit(1)

    groups = [g for g in match.groups() if g is not None]
    _, first, _, second, _ = groups

    if first.startswith(PREM_MARKERS[0]):
        first = '{prem}'
        second = '{hypo}'
        subst = r'\1' + first + r'\3' + second + r'\5'
    elif first.startswith(HYPO_MARKERS[0]):
        second = '{prem}'
        first = '{hypo}'
        subst = r'\6' + first + r'\8' + second + r'\10'
    else:
        print("Extraction does not start with prem nor hypo markers!")
        print(first)
        print(pattern)
        print(groups)
        exit(1)

    return regex.sub(subst, pattern)


def precompile_extraction_regex():
    global PREM_MARKERS
    global HYPO_MARKERS

    template = '(.*)({}[^{}]+{})(.*)({}[^{}]+{})(.*)'

    prem_first = template.format(
        re.escape(PREM_MARKERS[0]),
        re.escape(PREM_MARKERS[1][0]),
        re.escape(PREM_MARKERS[1]),
        re.escape(HYPO_MARKERS[0]),
        re.escape(HYPO_MARKERS[1][0]),
        re.escape(HYPO_MARKERS[1])
    )
    hypo_first = template.format(
        re.escape(HYPO_MARKERS[0]),
        re.escape(HYPO_MARKERS[1][0]),
        re.escape(HYPO_MARKERS[1]),
        re.escape(PREM_MARKERS[0]),
        re.escape(PREM_MARKERS[1][0]),
        re.escape(PREM_MARKERS[1])
    )

    return re.compile('{}|{}'.format(prem_first, hypo_first))


def main(args: argparse.Namespace):
    '''
    - unify multiple text files with instantiated patterns
    - identify which is premise and which is hypothesis; and mark them with {prem}/{hypo}
    - write everything out in a single file
    '''

    regex = precompile_extraction_regex()
    patterns = []
    for pat_file in tqdm(args.inst_pat_file):
        with open(pat_file) as f:
            for line in f:
                pattern = clean_instantiated_pattern(
                    line.strip(), regex
                )
                patterns.append(pattern)

    with open(args.unified_pattern_file, 'w') as fout:
        for p in patterns:
            print(p, file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inst_pat_file', nargs='+')
    parser.add_argument('unified_pattern_file')
    args = parser.parse_args()

    main(args)
