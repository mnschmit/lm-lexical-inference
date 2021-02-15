import argparse


def str_float_pair(pair):
    s, f = pair.split(',')
    return s, float(f)


def main(args):
    pred = {}
    truth = []
    for fname, thr in args.score_file_with_threshold:
        with open(fname) as f:
            lines = f.readlines()
            sample = lines[args.sample_no+1].strip()

        score, label = sample.split('\t')
        truth.append(int(label))
        pred[fname] = 1 if float(score) > thr else 0
    assert all([truth[i] == truth[i-1] for i in range(len(truth))])

    print('Truth :', truth[0])
    for fname, pred_value in pred.items():
        print(fname, ':', pred_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_no', type=int)
    parser.add_argument('score_file_with_threshold',
                        type=str_float_pair, nargs='+')
    args = parser.parse_args()
    main(args)
