#!/usr/bin/env python
import csv, copy, numpy, random, cPickle, datetime
import matplotlib.pyplot as pyplot
from mapping import Mapping, Change, ChangeNoStem
from featuredict import FeatureDict
from GEN import Input, Gen, DeterministicGen
from CON import Con, Markedness, MarkednessAligned, Faithfulness
from matplotlib.backends.backend_pdf import PdfPages
#THINGS TO WATCH OUT FOR:
    # delete the saved input file if you change the input making code
    # implement tableau making if you give it a file with ungrammatical mappings
#TODO make non_boundaries dict in FeatureDict and make it read "boundary" from feature names
#TODO Faithfulness: delete a change that's being reversed?
#TODO time random.sample vs numpy.random.sample
#TODO copy problem in diff_ngrams
# keep in mind: tiers can mess up alignment

#TODO think about using losers to help guide constraint induction. if you make a constraint, you want it to privilege cw over gw, but you
# also want cw to win over losers. could this help?
#TODO look at Jason Riggle's GEN and think about using CON to make GEN.

#TODO thanks to Gaja: maybe randomly choose between inducing markedness and faithfulness,
# start by making general constraints, save the most specified version of it,
# when making the same or the opposite, compare to most specified version and
# figure out how to make it more specific in a helpful way. most specified
# version should probably include 1 segment on either side.

#Profiler:
# import cProfile, learner, pstats
# cProfile.run("learner.Learn('feature_chart4.csv', ['input5.csv'], processes = '[self.change_feature_value]')", 'learnerprofile.txt')
# p = pstats.Stats('learnerprofile.txt')
# p.sort_stats('cumulative').print_stats(30)

# tests:
    # different tier frequencies
    # with and without stem constraints
    # with and without zipfian distribution in constraint making
    # random gen vs deterministic gen

class HGGLA:
    def __init__(self, learning_rate, feature_dict, aligned, tier_freq, induction_freq):
        """Takes processed input and learns on it one tableau at a time.
        The constraints are updated by the difference in violation vectors
        between the computed winner and the desired winner,
        multiplied by the learning rate."""
        self.learning_rate = learning_rate
        self.con = Con(feature_dict, tier_freq, aligned)
        self.induction_freq = induction_freq

    def evaluate(self, tableau):
        """Use constraints to find mappings violations
        and constraint weights to find mappings harmony scores.
        From harmony scores, find and return the mapping predicted to win."""
        computed_winner = None
        grammatical_winner = None
        correct = None
        harmonies = []
        for mapping in tableau:
            self.con.get_violations(mapping)
            mapping.harmony = numpy.dot(self.con.weights, mapping.violations)
            harmonies.append(mapping.harmony)
            if mapping.grammatical:
                grammatical_winner = mapping
        highest_harmony = max(harmonies)
        computed_winner = [mapping for mapping in tableau if mapping.harmony == highest_harmony]
        try:
            correct = True if grammatical_winner.harmony == highest_harmony else False
        except AttributeError: # grammatical_winner is None because this is a test
            pass
        return (grammatical_winner, computed_winner, correct)

    def update(self, grammatical_winner, computed_winner):
        difference = grammatical_winner.violations - computed_winner.violations
        assert len(difference) == len(self.con.constraints) + 1
        assert difference[0] == 0
        self.con.weights += difference * self.learning_rate

    def train(self, inputs):
        self.errors = []
        self.error_numbers = []
        for i, tableau in enumerate(inputs):
            random.shuffle(tableau)
            self.train_tableau(tableau, i)
        return (self.errors, self.error_numbers)

    def train_tableau(self, tableau, i):
        (grammatical_winner, computed_winner, correct) = self.evaluate(tableau)
        if correct:
            if len(computed_winner) != 1:
                computed_winner.remove(grammatical_winner)
                self.con.induce([grammatical_winner, computed_winner[0]])
                self.errors.append(''.join([str(i), ': ', str(grammatical_winner), '~t~', str([str(cw) for cw in computed_winner])]))
                self.error_numbers.append(i)
                #self.train_tableau(tableau) # if you use this, think about error counting
        else:
            if numpy.random.random() <= self.induction_freq:
                self.con.induce([grammatical_winner, computed_winner[0]])
                #self.train_tableau(tableau)
            else:
                self.update(grammatical_winner, computed_winner[0])
            self.errors.append(''.join([str(grammatical_winner), '~', str(computed_winner[0])]))

    def test(self, tableau):
        computed_winner = None
        harmonies = []
        for mapping in tableau:
            self.con.get_violations(mapping)
            mapping.harmony = numpy.dot(self.con.weights, mapping.violations)
            harmonies.append(mapping.harmony)
            highest_harmony = max(harmonies)
            computed_winner = [mapping for mapping in tableau if mapping.harmony == highest_harmony]
        computed_winner = computed_winner[0] if len(computed_winner) == 1 else random.choice(computed_winner)
        return computed_winner

class Learn:
    def __init__(self, feature_chart, input_file, remake_input = False, num_negatives = 15, max_changes = 5,
                 processes = '[self.change_feature_value]', #'[self.delete, self.metathesize, self.change_feature_value, self.epenthesize]',
                 epenthetics = ['e', '?'], learning_rate = 0.1, num_trainings = 3,
                 aligned = True, tier_freq = .2, stem = True, induction_freq = .1, constraint_parts = ['voi', '+word', 'round', 'back']):
        # input parameters
        self.feature_dict = FeatureDict(feature_chart)
        self.input_file = input_file
        self.remake_input = remake_input
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = processes
        self.epenthetics = epenthetics
        self.stem = stem

        # algorithm parameters
        self.learning_rate = learning_rate
        self.num_trainings = num_trainings
        self.aligned = aligned
        self.tier_freq = tier_freq
        self.induction_freq = induction_freq

        # input data
        self.all_input = None
        self.train_input = None
        self.test_input = None

        # output data
        time = str(datetime.datetime.now())
        self.figs = PdfPages('Output-' + time + '.pdf')
        self.report = 'Output-' + time + '.txt'
        self.training_errors = []
        self.testing_errors = []
        self.num_constraints = []
        self.constraint_parts = constraint_parts
        with open(self.report, 'a') as f:
            f.write('\n'.join(['Feature Chart: ' + feature_chart, 'Input File: ' + input_file, '# Ungrammatical Candidates Generated: ' + str(num_negatives),
                               'Max Changes to Candidates: ' + str(max_changes), 'GEN processes: ' + processes, 'Epenthetic Segments: ' + str(epenthetics),
                               'Learning Rate: ' + str(learning_rate), 'Aligned Markedness Constraints: ' + str(aligned), 'Frequency of Tier Constraints: ' +
                               str(tier_freq), 'Frequency of Induction Upon Error: ' + str(induction_freq)]))

    def make_input(self):
        """Use Input class to convert input file to data structure or access previously saved data structure."""
        inputs = Input(self.feature_dict, self.input_file, self.remake_input, self.num_negatives,
                       self.max_changes, self.processes, self.epenthetics, self.stem)
        self.all_input = inputs.allinputs

    def divide_input(self):
        """Choose training set and test set."""
        test_size = int(len(self.all_input)/10)
        numpy.random.shuffle(self.all_input)
        self.test_input = self.all_input[:test_size]
        self.train_input = self.all_input[test_size:test_size * 2]

    def refresh(self):
        for inputs in (self.train_input, self.test_input):
            for tableau in inputs:
                for mapping in tableau:
                    mapping.violations = numpy.array([1])
                    mapping.harmony = None
        self.alg.con.constraints = []
        self.alg.con.weights = numpy.array([0])

    def run(self, i):
        """Initialize HGGLA and do all training and testing iterations for this run."""
        self.alg = HGGLA(self.learning_rate, self.feature_dict, self.aligned, self.tier_freq, self.induction_freq)
        if i:
            self.refresh()
        for log in [self.training_errors, self.num_constraints, self.testing_errors]:
            log.append([])
            assert log[i] == []
        for j in range(self.num_trainings):
            (training_errors, num_constraints) = self.train_HGGLA(j)
            self.training_errors[i].append(training_errors)
            self.num_constraints[i].append(num_constraints)
            self.testing_errors[i].append(self.test_HGGLA(j))
        #self.training_runs.append(self.training_errors)
        #self.testing_runs.append(self.testing_errors)
        #self.num_constraints_runs.append(self.num_constraints)
        self.plot_constraints()
        self.check_constraints()

    def train_HGGLA(self, i):
        """Do one iteration through the training data."""
        errors, error_numbers = self.alg.train(self.train_input)
        #self.training_errors.append(float(len(errors))/float(len(self.train_input)))
        constraints_added = self.alg.con.constraints[self.num_constraints[-1][-1]:] if i else self.alg.con.constraints
        #self.num_constraints.append(len(self.alg.con.constraints))
        with open(self.report, 'a') as f:
            f.write(''.join(['\n\nErrors in training #', str(i), ': ', str(len(errors)), '\n', '\n\n'.join([error for error in errors])]))
            f.write(''.join(['\n\nConstraints added in training #', str(i), ': ',
                             str(len(constraints_added)), '\n', '\n'.join([str(c) for c in constraints_added])]))
        return (float(len(errors))/float(len(self.train_input)), len(self.alg.con.constraints))

    def test_HGGLA(self, i):
        """Do one iteration through the testing data."""
        errors = []
        for tableau in self.test_input:
            winner = self.alg.test(tableau)
            if winner.grammatical == False:
                errors.append(str(winner))
        #self.testing_errors.append(float(len(errors))/float(len(self.test_input)))
        with open(self.report, 'a') as f:
            f.write(''.join(['\n\nErrors in testing #', str(i), ': ', str(len(errors)), '\n', '\n\n'.join([error for error in errors])]))
        return float(len(errors))/float(len(self.test_input))

    def test_parameter(self, parameter, values):
        """If parameter is an input parameter, redo input for each value of the
        parameter. If parameter is an algorithm parameter, make and divide the
        input once, and then run the algorithm with each value of the
        parameter."""
        if parameter in ['self.feature_dict', 'self.input_file', 'self.num_negatives', 'self.max_changes', 'self.processes', 'self.epenthetics', 'self.stem']:
            param = eval(parameter)
            for i, value in enumerate(values):
                self.param = value
                self.remake_input = True
                self.make_input()
                self.divide_input()
                with open(self.report, 'a') as f:
                    f.write(' '.join(['\n\n\n--------', parameter, '=', str(value), '--------']))
                self.run(i)
        elif parameter in ['self.learning_rate', 'self.num_trainings', 'self.aligned', 'self.tier_freq', 'self.induction_freq']:
            param = eval(parameter)
            self.make_input()
            self.divide_input()
            for i, value in enumerate(values):
                self.param = value
                with open(self.report, 'a') as f:
                    f.write(' '.join(['\n\n\n--------', parameter, '=', str(value), '--------']))
                self.run(i)
        else:
            raise AssertionError, 'Update parameter lists.'
        print 'tested ', parameter, 'on ', values, '\n'
        print 'error percentage on last test of each run', [run[-1] for run in self.testing_errors], '\nnumber of constraints', [run[-1] for run in self.num_constraints]
        self.plot_errors(parameter = parameter, values = values)
        self.figs.close()

    def test_performance(self, num_runs = 5):
        """Make the input once, and then run several times with a new division
        into training and testing sets each time. Get average accuracy
        values."""
        self.make_input()
        for i in range(num_runs):
            self.divide_input()
            with open(self.report, 'a') as f:
                f.write('\n\n\n--------Run ' + str(i) + '--------')
            self.run(i)
        print 'ran program ', num_runs, ' times'
        print 'error percentage on last test of each run', [run[-1] for run in self.testing_errors], '\nnumber of constraints', [run[-1] for run in self.num_constraints]
        self.plot_errors()
        self.figs.close()

    def plot_constraints(self):
        """Plot the 20 highest weighted constraints and write all of them to a
        file with their weights, in descending order of weight."""
        constraints = [str(c) for c in self.alg.con.constraints]
        num_con = len(constraints)
        constraint_list = zip(self.alg.con.weights, constraints)
        constraint_list.sort(reverse = True)
        self.constraint_lines = [str(w) + '\t' + c for (w, c) in constraint_list]
        with open(self.report, 'a') as f:
            f.write('\n\nTotal Constraints, Sorted by Weight\n')
            f.write('\n'.join(self.constraint_lines))
        constraint_list = zip(*constraint_list)
        #ind = numpy.arange(num_con)
        length = 20 if num_con > 20 else num_con
        ind = numpy.arange(length)
        height = .35
        #colors = ['r' if isinstance(c, Faithfulness) else 'y' for c in self.alg.con.constraints]
        pyplot.barh(ind, constraint_list[0][0:length]) #, color = colors
        pyplot.xlabel('Weights')
        pyplot.title('Constraint Weights')
        pyplot.yticks(ind+height/2., constraint_list[1][0:length])
        pyplot.subplots_adjust(left = .5) # make room for constraint names
        self.figs.savefig()
        pyplot.clf()

    def plot_errors(self, parameter = None, values = None):
        """Make three plots, one for training, one for testing, and one for number of constraints. Each has
        iterations on the x axis and error percentage on the y axis. Plot a line
        for each run."""
        pyplot.subplots_adjust(left = .15)
        for item in [self.training_errors, self.testing_errors, self.num_constraints]:
            plots = []
            for run in item:
                plots.append(pyplot.plot(run))
            pyplot.xlabel('Iteration')
            pyplot.xticks(numpy.arange(self.num_trainings))
            if item != self.num_constraints:
                pyplot.ylim(-.1, 1.1)
                pyplot.ylabel('Percent of Inputs Mapped to Incorrect Outputs')
                pyplot.yticks(numpy.arange(-.1, 1.1, .1))
                kind = 'Training' if item == self.training_errors else 'Testing'
                pyplot.title(kind + ' Error Rates')
            else:
                pyplot.ylabel('Number of Constraints')
                pyplot.title('Running Count of Constraints')
            labels = [parameter + ' = ' + str(value) for value in values] if parameter else range(len(item))
            pyplot.legend(plots, labels, loc = 0)
            #pyplot.show()
            self.figs.savefig()
            pyplot.clf()

    def check_constraints(self):
        constraints_found = []
        for line in self.constraint_lines:
            for part in self.constraint_parts:
                if part in line:
                    constraints_found.append(line)
                    break
        #print constraints_found
        with open(self.report, 'a') as f:
            f.write('\n\nConstraints With Expected Features\n' + '\n'.join(constraints_found))

class CrossValidate(Learn):
    """Train the algorithm on every possible set of all but one data point
    and test on the leftover data point.
    Look at the average accuracy across tests."""

    def run_HGGLA(self, inputs):
        tableaux = self.make_tableaux(inputs)
        for i, tableau in enumerate(tableaux):
            self.refresh_input(tableaux)
            self.refresh_con()
            assert self.alg.con.constraints == []
            training_set = tableaux[:i] + tableaux[i + 1:]
            for i in range(self.num_trainings):
                errors = self.alg.train(training_set)
                self.training_errors.append(sum(errors))
            # test
            desired = None
            test_tableau = []
            for mapping in tableau:
                if mapping.grammatical:
                    desired = mapping.sr
                map_copy = copy.deepcopy(mapping)
                map_copy.grammatical = 'test'
                test_tableau.append(map_copy)
            for m in test_tableau:
                assert m.harmony == None
                assert m.violations == numpy.array([1])
            if self.alg.test(test_tableau).sr == desired:
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)

if __name__ == '__main__':
    import os
    import sys
    #localpath = os.getcwd() + '/' + '/'.join(sys.argv[0].split('/')[:-1]) #don't use
    localpath = '/'.join(sys.argv[0].split('/')[:-1])
    os.chdir(localpath)
    #learner = Learn('feature_chart4.csv', 'input6.csv', processes = '[self.change_feature_value]', max_changes = 5, num_negatives = 20, induction_freq = .5)
    learner = Learn('TurkishFeaturesWithNA.csv', 'TurkishInput4.csv', num_trainings = 5, learning_rate = .1,
                         max_changes = 5, num_negatives = 15, tier_freq = 0, processes = '[self.change_feature_value]')
    #TurkishInput2 has the ~ inputs taken out, the variable inputs taken out, and deletion taken out.
    #TurkishInput1 is the same but deletion is still in.
    #same pattern for test files
    #TurkishInput3 is TurkishInput2 plus TurkishTest2
    #TurkishInput4 is TurkishInput3 with all underlying suffix vowels changed to i, and appropriate changes added.
    #learner.test_performance(2)
    #learner.test_parameter('self.learning_rate', [.01, .05, .1, .2, .3, .4, .5])
    #learner.test_parameter('self.induction_freq', [0, .1, .2, .3, .4, .5, .6, .7, .8])
    #learner.test_parameter('self.max_changes', [2, 5, 10])
    learner.test_parameter('self.tier_freq', [0, .1, .2, .3, .4, .5])
