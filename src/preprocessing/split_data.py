import argparse
import csv
from sklearn.model_selection import train_test_split


def process_levy(args: argparse.Namespace):
    lines, targets = [], []
    with open(args.data_file) as f:
        for line in f:
            hypo, prem, target = line.strip().split('\t')
            lines.append(line)
            targets.append(target == 'True')

    lines1, lines2 = train_test_split(
        lines, stratify=targets,
        random_state=args.seed, test_size=args.test_size
    )

    with open(args.split1_file, 'w') as fout1, open(args.split2_file, 'w') as fout2:
        fout1.writelines(lines1)
        fout2.writelines(lines2)


def process_sherliic(args: argparse.Namespace):
    with open(args.data_file) as f:
        cr = csv.reader(f)
        next(cr)  # headers
        targets = [row[18] == 'yes' for row in cr]
        f.seek(0)
        header_line = next(f)
        lines = f.readlines()
    assert len(lines) == len(targets)

    lines1, lines2 = train_test_split(
        lines, stratify=targets,
        random_state=args.seed, test_size=args.test_size
    )
    with open(args.split1_file, 'w') as fout1, open(args.split2_file, 'w') as fout2:
        fout1.write(header_line)
        fout1.writelines(lines1)
        fout2.write(header_line)
        fout2.writelines(lines2)


def main(args: argparse.Namespace):
    if args.levy:
        process_levy(args)
    else:
        process_sherliic(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('split1_file')
    parser.add_argument('split2_file')
    parser.add_argument('--seed', type=int, default=47110815)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--levy', action='store_true')
    args = parser.parse_args()
    main(args)
