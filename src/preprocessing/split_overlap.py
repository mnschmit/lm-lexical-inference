import argparse
from typing import List


def extract_relations(fname: str) -> List[str]:
    rels = []
    with open(fname) as f:
        for line in f:
            hypo, prem, cls = line.strip().split('\t')
            rels.append(hypo.split(',')[1])
            # rels.append(prem.split(',')[1])
    return rels


def main(args):
    train_relations = set(extract_relations(args.train_file))
    dev_relations = set(extract_relations(args.dev_file))
    common = train_relations & dev_relations

    print(common)
    print(len(common))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('dev_file')
    args = parser.parse_args()

    main(args)
