import argparse
from sklearn.metrics import precision_recall_curve


def main(args):
    scores, labels = [], []
    with open(args.score_file) as f:
        next(f)  # headers
        for line in f:
            score, label = line.strip().split('\t')
            scores.append(float(score))
            labels.append(int(label))

    prec_rec_thr = precision_recall_curve(labels, scores, pos_label=1)
    thr2scores = {}
    for p, r, t in zip(*prec_rec_thr):
        f1 = 2 * p * r / (p+r)
        thr2scores[t] = (p, r, f1)
    best_thr = max(thr2scores.keys(), key=lambda t: thr2scores[t][2])

    print(
        "Best threshold is {} with P/R/F1 scores of {}/{}/{}.".format(
            best_thr, *thr2scores[best_thr]
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score_file')
    args = parser.parse_args()

    main(args)
