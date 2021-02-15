import argparse
import csv


def with_examples(sent_tuple, A_ex, B_ex):
    left, middle, right, end = sent_tuple
    if left.strip().endswith('[A]'):
        left = A_ex
        right = B_ex
    else:
        right = A_ex
        left = B_ex
    return left, middle, right, end


def file_thr_pair(arg: str):
    fn, thr = arg.split(',')
    return fn, float(thr)


def load_data_levy_holt(data_file):
    data = []
    with open(data_file) as f:
        for line in f:
            hypo, prem, cls = line.strip().split('\t')
            # hypo = hypo.split(',')
            # prem = prem.split(',')
            data.append((prem, hypo, cls == 'True'))
    return data


def load_data_sherliic(data_file):
    data = []
    with open(data_file) as f:
        r = csv.reader(f)
        next(r)  # headers
        for row in r:
            A_example = row[15].split(' / ')[0]
            B_example = row[16].split(' / ')[0]
            prem = with_examples(row[5:9], A_example, B_example)
            hypo = with_examples(row[9:13], A_example, B_example)
            data.append((prem, hypo, row[17] == 'yes'))
    return data


def load_scores(score_filename):
    scores = []
    with open(score_filename) as f:
        next(f)  # headers
        for line in f:
            score, label = line.strip().split('\t')
            scores.append(float(score))
    return scores


def search_errors(data, scores, thr):
    fp, fn = [], []
    for line_no, (inst, score) in enumerate(zip(data, scores)):
        if score > thr:
            if not inst[-1]:
                fp.append((inst, score, line_no))
        else:
            if inst[-1]:
                fn.append((inst, score, line_no))

    fp.sort(key=lambda p: p[1], reverse=True)
    fn.sort(key=lambda p: p[1], reverse=False)

    return fp, fn


def get_errors_from_system(data, score_fn, thr):
    scores = load_scores(score_fn)
    return search_errors(data, scores, thr)


def filter_errors(errors, to_be_excluded):
    filtered_errors = []
    for dp1, model_score1, line_no1 in errors:
        add_it = True
        for dp2, model_score2, line_no2 in to_be_excluded:
            if dp2 == dp1:
                add_it = False
                break
        if add_it:
            filtered_errors.append((dp1, model_score1, line_no1))
    return filtered_errors


def main(args):
    if args.levy_holt:
        data = load_data_levy_holt(args.data_file)
    else:
        data = load_data_sherliic(args.data_file)

    fp, fn = get_errors_from_system(data, *args.score_file_with_threshold)

    if args.exclude_errors_from is not None:
        ex_fp, ex_fn = get_errors_from_system(data, *args.exclude_errors_from)
        fp = filter_errors(fp, ex_fp)
        fn = filter_errors(fn, ex_fn)

    print('=== False Positives ===')
    for e in fp:
        print(e)
    print('\n=== False Negative ===')
    for e in fn:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score_file_with_threshold', type=file_thr_pair)
    parser.add_argument('data_file')
    parser.add_argument('--exclude_errors_from',
                        type=file_thr_pair, default=None)
    parser.add_argument('--levy_holt', action='store_true')
    args = parser.parse_args()

    main(args)
