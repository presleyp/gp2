import learner
l = learner.Learn('TurkishFeaturesWithNA.csv',
                    'TurkishInput4NoVHExceptions.csv',
                    #'TurkishSmallInput.csv',
                    num_trainings = 5,
                    #num_trainings = 1,
                    gen_type = 'deterministic',
                    learning_rate = .2,
                    induction_freq = .5,
                    stem = 0,
                    tier_freq = 0,
                    decay_rate = 0,
                    #remake_input = True,
                    report_id = 'make-input')
l.make_input()
