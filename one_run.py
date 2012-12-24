# usage: python one_run.py 0 0.4 1-0-40

import learner, sys, csv

def matrix_to_dataframe(filename, i, j, matrix):
    with open(filename, 'w') as f:
        cwrite = csv.writer(f)
        for row in matrix:
            for k, number in enumerate(row):
                cwrite.writerow([number, i, j, k])


def run_once(stem_val, tier_val, decay, run_id):
    l = learner.Learn('TurkishFeaturesWithNA.csv',
                    'TurkishInput4NoVHExceptions.csv',
                    #'TurkishSmallInput.csv',
                    num_trainings = 5,
                    gen_type = 'deterministic',
                    learning_rate = .1,
                    induction_freq = .5,
                    stem = stem_val,
                    tier_freq = tier_val,
                    decay_rate = decay,
                    #remake_input = True,
                    report_id = run_id)
    l.make_input()
    l.divide_input()
    outputs = l.run()
    for (name, matrix) in zip(['training', 'testing', 'constraints'], outputs):
        filename = name + '_stem_tier_' + run_id + '.csv'
        matrix_to_dataframe(filename, stem_val, tier_val, matrix)

if __name__ == '__main__':
    stem_val = float(sys.argv[1])
    tier_val = float(sys.argv[2])
    decay = float(sys.argv[3])
    run_id = sys.argv[4]
    run_once(stem_val, tier_val, decay, run_id)
