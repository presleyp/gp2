def matrix_to_dataframe(filename, i, j, matrix):
    with open(filename, 'a') as f:
        cwrite = csv.writer(f)
        for row in matrix:
            for k, number in enumerate(row):
                cwrite.writerow([number, i, j, k])

def run_experiment(time, i, j):
    print 'experiment with ', i, ' and ', j
    l = Learn('TurkishFeaturesWithNA.csv', 'TurkishInput4.csv', num_trainings = 5, gen_type = 'deterministic', learning_rate = i, induction_freq = j, stem = True, tier_freq = 0.25)
    outputs = l.test_performance(20)
    for (name, matrix) in zip(['training ', 'testing ', 'constraints '], outputs):
        filename2 = time + name + '_lr_induction.csv'
        matrix_to_dataframe(filename2, i, j, matrix)

time = datetime.datetime.now()
time = time.strftime('%Y-%m-%d-%H:%M:%S')
if len(sys.argv) < 3:
    for i in [0.01, 0.1, 0.25]:
        for j in [0.1, 0.5, 0.9]:
            run_experiment(time, i, j)
else:
    i = float(sys.argv[1])
    j = float(sys.argv[2])
    run_experiment(time, i, j)

