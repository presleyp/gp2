#!/usr/bin/env python
import csv, copy, random, numpy
#TODO get numpy working with python3
#TODO figure out how to represent tiers
#TODO improve change logging
#TODO change input files to have empty changelog for faithful cands - if that creates problems, change generated changelogs to say 'none'

class Input:
    """Give input in the form of a csv file where each line is a mapping, with 0 or 1 for ungrammatical or grammatical, then underlying form
    in segments, then surface form in segments, then semicolon-delimited list of changes from underlying form to surface form.
    Ungrammatical mappings are optional; if you include them, the second line must be ungrammatical."""
    def __init__(self, feature_dict, infile, num_negatives, max_changes, processes, epenthetics):
        """Convert lines of input to mapping objects.
        Generate ungrammatical input-output pairs if they were not already present in the input file."""
        self.feature_dict = feature_dict
        self.gen_args = [num_negatives, max_changes, processes, epenthetics]
        self.allinputs = self.make_input(infile)

    def make_input(self, infile):
        """Based on file of lines of the form "1 underlyingform surfaceform changes"
        create Mapping objects with those attributes.
        Make words into lists of segments,
        segments into dictionaries of feature values,
        and changes into a list of strings.
        Create ungrammatical mappings if not present."""
        allinputs = []
        ungrammatical_included = False # could change to be passed as arg, but this is intuitive for me
        with open(infile, 'r') as f:
            fread = list(csv.reader(f))
            if fread[1][0] == '0':
                ungrammatical_included = True
            for (grammatical, ur, sr, changes) in fread:
                mapping = Mapping(grammatical, ur, sr, changes, self.feature_dict)
                mapping.to_data()
                allinputs.append(mapping)
                if ungrammatical_included == False:
                    gen = Gen(self.feature_dict, *self.gen_args)
                    negatives = gen.ungrammaticalize(mapping)
                    allinputs += negatives
        return allinputs

class Gen:
    """Generates input-output mappings."""
    def __init__(self, feature_dict, num_negatives, max_changes, processes, epenthetics):
        self.feature_dict = feature_dict
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = processes
        self.epenthetics = epenthetics

    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly ungrammatical mappings with the same input."""
        negatives = []
        i = 0
        while len(negatives) < self.num_negatives:
            i += 1
            new_mapping = Mapping(False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), [], self.feature_dict)
            for j in range(random.randint(0, self.max_changes)):       # randomly pick number of processes
                process = random.choice(eval(self.processes))          # randomly pick process
                process(new_mapping)
            # Don't add a mapping if it's the same as the grammatical one.
            if new_mapping.sr == mapping.sr:
                continue
            # Don't add a mapping if its sr has a segment that isn't in the
            # inventory
            try:
                #print new_mapping
                self.feature_dict.get_segments(new_mapping.sr)
            except IndexError:
                continue
            # Don't add a mapping if it's the same as a previously
            # generated one.
            duplicate = False
            for negative in negatives:
                if new_mapping.sr == negative.sr:
                    duplicate = True
                    break
            if duplicate == False:
                negatives.append(new_mapping)
        return negatives

    def epenthesize(self, mapping):
        """Map a ur to an sr with one more segment."""
        epenthetic_segment = random.choice(self.epenthetics)
        epenthetic_features = self.feature_dict.get_features(epenthetic_segment)
        locus = random.randint(0, len(mapping.sr))
        mapping.sr.insert(locus, epenthetic_features)
        mapping.changes.append('epenthesize ' + epenthetic_segment)

    def delete(self, mapping):
        """Map a ur to an sr with one less segment."""
        if len(mapping.sr) > 1:
            locus = random.randint(0, len(mapping.sr) - 1)
            deleted = mapping.sr.pop(locus)
            try:
                deleted = self.feature_dict.get_segment(deleted)
            except IndexError:
                pass
            finally:
                mapping.changes.append('delete ' + str(deleted))

    def metathesize(self, mapping):
        """Map a ur to an sr with two segments having swapped places."""
        if len(mapping.sr) > 1:
            locus = random.randint(0, len(mapping.sr) - 2)
            moved_left = mapping.sr[locus + 1]
            moved_right = mapping.sr.pop(locus)
            mapping.sr.insert(locus + 1, moved_right)
            try:
                moved_right = self.feature_dict.get_segment(moved_right)
                moved_left = self.feature_dict.get_segment(moved_left)
            except IndexError:
                pass
            finally:
                mapping.changes.append('metathesize ' + str(moved_right) + ' and ' + str(moved_left))

    # I want to change feature values rather than change from one segment to
    # another so that the probability of changing from t to d is higher than the
    # probability of changing from t to m. But this means it's
    # possible to change to a set of feature values that maps to no
    # segment. I could allow this and manage the segmental inventory with
    # unigram constraints, but it seems odd to allow segments that aren't even
    # physically possible. So I'll at least filter those out (and pretend
    # phonetics is doing it, perhaps); for now I'll also filter out segments not
    # in the inventory, but in the future it's may be better to change that.
    def change_feature_value(self, mapping):
        """Map a ur to an sr with the value of a feature of a segment changed."""
        locus = random.randint(0, len(mapping.sr) - 1)
        segment = mapping.sr[locus]
        original_segment = copy.deepcopy(segment)
        feature = random.choice(segment.keys()) # wrap in list() for python3
        segment[feature] = '1' if segment[feature] == '0' else '0'
        value = segment[feature]
        try:
            original_segment = self.feature_dict.get_segment(original_segment)
        except IndexError:
            pass
        finally:
            mapping.changes.append('change ' + str(feature) + ' in ' + str(original_segment) + ' to ' + str(value))

class Mapping:
    def __init__(self, grammatical, ur, sr, changes, feature_dict):
        """Each input-output mapping is an object with attributes: grammatical (is it grammatical?),
        ur (underlying form), sr (surface form), changes (operations to get from ur to sr),
        violations (of constraints in order), and harmony (violations times constraint weights).
        ur and sr are lists of segments, which are dictionaries of feature-value pairs."""
        self.grammatical = grammatical
        self.ur = ur
        self.sr = sr
        self.changes = changes
        self.violations = []
        self.harmony = None
        self.feature_dict = feature_dict

    def to_data(self):
        self.grammatical = bool(int(self.grammatical))
        self.ur = [self.feature_dict.get_features(segment) for segment in self.ur]
        self.sr = [self.feature_dict.get_features(segment) for segment in self.sr]
        self.changes = [] if self.changes == 'none' else self.changes.split(';')

    def __str__(self):
        ur = ''.join(self.feature_dict.get_segments(self.ur))
        sr = ''.join(self.feature_dict.get_segments(self.sr))
        return ', '.join([str(self.grammatical), ur, sr, str(self.changes)])

class FeatureDict:
    def __init__(self, feature_chart):
        """Make a dictionary of dictionaries like {segment:{feature:value, ...} ...}.
        This will be used to determine the features of the input segments.
        Also find the number of features in the chart, which will be used in constraint induction."""
        self.fd = {}
        with open(feature_chart, 'r') as fc:
            fcd = csv.DictReader(fc)
            for line in fcd:
                seg = line.pop('segment')
                self.fd[seg] = line

    def get_features(self, segment):
        features = copy.deepcopy(self.fd[segment])
        return features

    def get_segments(self, word):
        segments = [self.get_segment(features) for features in word] # change iteritems to items for python3
        return segments

    def get_segment(self, features):
        segment = [k for k, v in self.fd.iteritems() if v == features][0]
        return segment
        #try:
            #segment = [k for k, v in self.fd.iteritems() if v == features][0]
            #return segment
        #except IndexError:
            #l = [k for k, v in self.fd.iteritems() if v == features]
            #print l

    #def notInInventory(self, word):
        #return False in map(lambda x: x in self.fd.values(), word)

class Con:
    def __init__(self, feature_dict):
        self.constraints = []
        self.weights = numpy.array([0]) # intercept weight
        key = random.choice(feature_dict.fd.keys()) # list() in python 3
        self.num_features = len(feature_dict.fd[key])

    def induce(self, computed_winner):
        """Makes one new markedness and one new faithfulness constraint for each pair in the training data,
        and initializes their weights."""
        # TODO make it create a m and a f constraint against things that exist
        # in the computed but not in the grammatical
        # and the reverse? gen won't create infinite goodness problem...
        self.constraints.append(Markedness(computed_winner.sr, self.num_features))
        if computed_winner.changes != []:
            self.constraints.append(Faithfulness(computed_winner.changes))
        added = len(self.constraints) - (len(self.weights) - 1) # weights will always be 1 longer because of intercept weight
        for i in range(added):
            self.weights = numpy.append(self.weights, random.random()) # could change to random low value

    def get_violations(self, mapping):
        mapping.violations = numpy.array([1]) # intercept
        mapping.violations = numpy.append(mapping.violations, numpy.array([constraint.get_violation(mapping)
                                                                           for constraint in self.constraints]))
#class Intercept:
    #def __init__(self):
        #pass

    #def get_violation(self):
        #return 1

class Markedness:
    def __init__(self, sr, num_features):
        """For a given output, make a constraint against one, two,
        or three adjacent natural classes. The constraint is a list of
        sets of feature-value tuples."""
        locus = random.randint(0, len(sr) - 1)
        self.constraint = []
        self.gram = 0
        for gram in range(random.randint(1, 3)):
            try:
                natural_class = random.sample(list(sr[locus].iteritems()), random.randint(1, num_features))
                self.constraint.append(set(natural_class))
                locus += 1
                self.gram += 1
            except IndexError: # sr ran out of segments
                break

    def get_violation(self, mapping):
        """Finds the number of places in the surface representation
        (including overlapping ones) that match the pattern of the constraint."""
        violation = 0
        for i in range(len(mapping.sr) - self.gram + 1):
            segments = mapping.sr[i:i + self.gram]
            assert len(segments) == self.gram, 'slice of sr is the wrong size'
            # if each segment in the slice is in the corresponding natural class
            # of the constraint, add a violation
            if False not in map(lambda x, y: set(x.iteritems()) >= y, segments, self.constraint):
                violation += 1
        return violation

class Faithfulness:
    def __init__(self, changes):
        self.constraint = random.choice(changes)

    def get_violation(self, mapping):
        """Finds the number of times the change referred to by the constraint occurs in the input-output pair."""
        return mapping.changes.count(self.constraint)

class HGGLA:
    def __init__(self, learning_rate, feature_dict):
        """Takes processed input and learns on it one tableau at a time.
        The constraints are updated by the difference in violation vectors
        between the computed winner and the desired winner,
        multiplied by the learning rate."""
        self.learning_rate = learning_rate
        self.constraints = Con(feature_dict)
        self.actual_winner = None

    def eval(self, tableau):
        """Use constraints to find mappings violations
        and constraint weights to find mappings harmony scores.
        From harmony scores, find and return the mapping predicted to win."""
        computed_winner = None
        while computed_winner == None:
            for i, mapping in enumerate(tableau):
                self.constraints.get_violations(mapping) #make sure this changes violations in place
                mapping.harmony = numpy.dot(self.constraints.weights, mapping.violations)
            highest_harmony = max([mapping.harmony for mapping in tableau])
            computed_winners = [mapping for mapping in tableau if mapping.harmony == highest_harmony]
            if len(computed_winners) > 1: # there's a tie
                #for i in range(len(computed_winners) - 1): # for inducing based on two forms
                    #self.constraints.induce(computed_winners[i], computed_winners[i + 1]) # though doesn't compare nonadjacent ties
                for cw in computed_winners:
                    self.constraints.induce(cw)
            else:
                #print 'found a cw'
                computed_winner = computed_winners[0]
        return computed_winner

    def train(self, inputs):
        for iteration in range(10): # learn from the data this many times
            differences = []
            for tableau in inputs: # learn one tableau at a time
                computed_winner = self.eval(tableau)
                print 'c winner', computed_winner
                grammatical_winner = None
                for mapping in tableau:
                    #print 'mg', mapping.grammatical
                    if mapping.grammatical == True:
                        grammatical_winner = mapping
                        #print 'g winner', grammatical_winner
                        break
                if grammatical_winner != computed_winner:
                    difference = grammatical_winner.violations - computed_winner.violations
                    self.constraints.weights += difference * self.learning_rate
                    differences.append(difference)
                    self.constraints.induce(computed_winner)
                else:
                    differences.append(0)
            if len(differences) != 0:
                print 'avg diff', numpy.mean(differences) # not sure if this is meaningful

    def test(self, inputs):
        winners = []
        for tableau in inputs: # probably only one of them, but open to other kinds of cross validation
            winners.append(self.eval(tableau))
        return winners

class CrossValidate:
    """Train the algorithm on every possible set of all but one data point
    and test on the leftover data point.
    Look at the average accuracy across tests."""
    def __init__(self, feature_chart, input_file, algorithm, learning_rate = 0.1, num_negatives = 5, max_changes = 10,
                 processes = '[self.delete, self.metathesize, self.change_feature_value, self.epenthesize]',
                 epenthetics = ['e', '?']):
        feature_dict = FeatureDict(feature_chart)
        inputs = Input(feature_dict, input_file, num_negatives, max_changes, processes, epenthetics)
        allinput = inputs.allinputs
        self.alg = algorithm(learning_rate, feature_dict)
        if algorithm == HGGLA:
            self.accuracy = self.HG_validate(allinput)
        else:
            self.accuracy = self.perceptron_validate(allinput)

    def make_tableaux(self, inputs):
        #print inputs
        urs = [mapping.ur for mapping in inputs if mapping.grammatical == True]
        tableaux = [[mapping for mapping in inputs if mapping.ur == item] for item in urs]
        return tableaux

    def HG_validate(self, inputs):
        tableaux = self.make_tableaux(inputs)
        self.accuracy = []
        for i, tableau in enumerate(tableaux):
            training_set = tableaux[:i] + tableaux[i + 1:]
            self.alg.train(training_set)
            # test
            desired = None
            test_tableau = []
            for mapping in tableau:
                if mapping.grammatical == True:
                    desired = mapping.sr
                map_copy = copy.deepcopy(mapping)
                map_copy.grammatical = 'test'
                test_tableau.append(map_copy)
            if self.alg.test([test_tableau])[0].sr == desired: # [0] because it returns a list of one element
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)
        return self.accuracy

    def perceptron_validate(self, inputs):
        self.accuracy = []
        for mapping in inputs:
            inputs.remove(mapping)
            answer = mapping.grammatical
            mapping.grammatical = None
            self.alg.train(inputs)
            if self.alg.test(mapping)[0] == answer:
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)
        return self.accuracy

if __name__ == '__main__':
    import os
    import sys
    #print sys.argv
    #print os.getcwd()
    #localpath = os.getcwd() + '/' + '/'.join(sys.argv[0].split('/')[:-1])
    localpath = '/'.join(sys.argv[0].split('/')[:-1])
    #print localpath
    os.chdir(localpath)
    xval1 = CrossValidate('feature_chart2.csv', 'input2.csv', HGGLA)
    #xval2 = CrossValidate('feature_chart2.csv', 'input3.csv', HGGLA)
    print( xval1.accuracy )
    #print( xval2.accuracy )
