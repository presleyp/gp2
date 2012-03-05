#!/usr/bin/env python
import csv, copy, random, numpy, itertools
#TODO get numpy working with python3
#TODO improve change logging - currently gives whole feature set, should give a sample, or just voc, cons, or son
#TODO implicational constraints
#TODO consider constraining GEN
#TODO graphs - error rate
#TODO consider adding lexically conditioned constraints
#TODO check M constraints for whether they really find a difference bw winners; sampling of features might mess this up

class Input:
    """Give input in the form of a csv file where each line is a mapping, with 0
    or 1 for ungrammatical or grammatical, then underlying form in segments,
    then surface form in segments, then semicolon-delimited list of changes from
    underlying form to surface form.  Ungrammatical mappings are optional; if
    you include them, the second line must be ungrammatical."""
    def __init__(self, feature_dict, infile, num_negatives, max_changes, processes,
        epenthetics):
        """Convert lines of input to mapping objects.  Generate
        ungrammatical input-output pairs if they were not already present in the
        input file."""
        self.feature_dict = feature_dict
        self.gen_args = [num_negatives, max_changes, processes, epenthetics]
        self.allinputs = self.make_input(infile)

    def make_input(self, infile):
        """Based on file of lines of the form "1,underlyingform,surfaceform,changes"
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
            for line in fread:
                mapping = Mapping(self.feature_dict, line)
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
        self.major_features = self.feature_dict.major_features
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = processes
        self.epenthetics = epenthetics

    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly ungrammatical mappings with the same input."""
        negatives = []
        while len(negatives) < self.num_negatives:
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
            for j in range(random.randint(0, self.max_changes)):       # randomly pick number of processes
                process = random.choice(eval(self.processes))          # randomly pick process
                process(new_mapping)
            # Don't add a mapping if it's the same as the grammatical one.
            if new_mapping.sr == mapping.sr:
                continue
            # Don't add a mapping if its sr has a segment that isn't in the
            # inventory
            try:
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

#TODO: change processes to accomodate matrices



    def epenthesize(self, mapping):
        """Map a ur to an sr with one more segment."""
        epenthetic_segment = random.choice(self.epenthetics)
        epenthetic_features = self.feature_dict.get_features(epenthetic_segment)
        locus = random.randint(0, len(mapping.sr))
        mapping.sr.insert(locus, epenthetic_features)
        new_change = ['epenthesize', self.major_features(epenthetic_features)]
        mapping.changes.append(' '.join([`item` for item in new_change]))

    def delete(self, mapping):
        """Map a ur to an sr with one less segment."""
        if len(mapping.sr) > 1:
            locus = random.randint(0, len(mapping.sr) - 1)
            deleted = mapping.sr.pop(locus)
            new_change = ['delete', self.major_features(deleted)]
            mapping.changes.append(' '.join([`item` for item in new_change]))

    def metathesize(self, mapping):
        """Map a ur to an sr with two segments having swapped places."""
        if len(mapping.sr) > 1:
            locus = random.randint(0, len(mapping.sr) - 2)
            moved_left = mapping.sr[locus + 1]
            moved_right = mapping.sr.pop(locus)
            mapping.sr.insert(locus + 1, moved_right)
            new_change = ['metathesize', self.major_features(moved_right), self.major_features(moved_left)]
            mapping.changes.append(' '.join([`item` for item in new_change]))

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
        major_features = self.major_features(segment)
        feature_index = random.randint(0, len(segment) - 1)
        segment[feature_index] = 1 if segment[feature_index] == 0 else 0
        value = segment[feature]
        # follow rule writing: change a to b in the environment of c
        new_change = ['change', feature_index, value, major_features]
        mapping.changes.append(' '.join([`item` for item in new_change]))

class Mapping:
    def __init__(self, feature_dict, line):
        """Each input-output mapping is an object with attributes: grammatical (is it grammatical?),
        ur (underlying form), sr (surface form), changes (operations to get from ur to sr),
        violations (of constraints in order), and harmony (violations times constraint weights).
        ur and sr are lists of segments, which are dictionaries of feature-value pairs."""
        self.grammatical = line[0]
        self.ur = line[1]
        self.sr = line[2]
        self.changes = line[3]
        self.violations = numpy.array([1]) # intercept
        self.harmony = None
        self.meaning = line[4] if len(line) == 5 else None
        self.feature_dict = feature_dict

    def to_data(self):
        self.grammatical = bool(int(self.grammatical))
        self.ur = self.feature_dict.get_features_word(self.ur)
        self.sr = self.feature_dict.get_features_word(self.sr)
        self.changes = [] if self.changes == 'none' else self.changes.split(';')
        for i in range(len(self.changes)):
            if 'change ' in self.changes[i]:
                changeparts = self.changes[i].split(' ')
                changeparts[3] = self.feature_dict.get_features_seg(changeparts[3])
                changeparts[3] = self.feature_dict.major_features(changeparts[3])
                self.changes[i] = ' '.join(changeparts)

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
            fcd = csv.reader(fc)
            for i, line in enumerate(fcd):
                if i > 0:
                    segment = line.pop(0)
                    self.num_features = len(line)
                    self.fd[segment] = numpy.array([int(value) for value in line])

    def get_features_seg(self, segment):
        return copy.copy(self.fd[segment])

    def get_features_word(self, word):
        features = [self.get_features_seg(segment) for segment in word]
        features = numpy.array(features)
        return features

    def get_segments(self, word):
        return [self.get_segment(features) for features in word]

    def get_segment(self, features):
        return [k for k, v in self.fd.iteritems() if v == features][0]

    def major_features(self, featureset):
        """Select only the first three features. These should be vocalic, consonantal, and sonorant."""
        return featureset[0:3]

class Con:
    def __init__(self, feature_dict, tier_freq):
        self.constraints = []
        self.weights = numpy.array([0]) # intercept weight
        self.num_features = feature_dict.num_features
        self.tier_freq = tier_freq

    def induce(self, winners):
        """Makes one new markedness and one new faithfulness constraint
        and initializes their weights, unless appropriate constraints
        cannot be found within 15 tries."""
        assert len(self.weights) == len(self.constraints) + 1
        self.make_constraint(Markedness, self.num_features, self.tier_freq, winners)
        self.make_constraint(Faithfulness, winners)
        new_weights = numpy.random.random(self.num_needed(self.weights))
        self.weights = numpy.append(self.weights, new_weights)
        assert len(self.weights) == len(self.constraints) + 1

    def make_constraint(self, constraint_type, *args):
        i = 0
        while i < 15:
            new_constraint = constraint_type(*args)
            i += 1
            if new_constraint not in self.constraints and new_constraint.constraint != None:
                self.constraints.append(new_constraint)
                break

    def get_violations(self, mapping):
        new_constraints = -self.num_needed(mapping.violations)
        if new_constraints < 0:
            new_violations = numpy.array([constraint.get_violation(mapping)
                                          for constraint in self.constraints[new_constraints:]])
            mapping.violations = numpy.append(mapping.violations, new_violations)
        assert len(mapping.violations) == len(self.constraints) + 1

    def num_needed(self, array):
        """Find number of constraints added since the array was updated,
        keeping in mind that the array has an intercept not included in
        the constraint set."""
        return len(self.constraints) - (len(array) - 1)

class Markedness:
    def __init__(self, num_features, tier_freq, winners):
        """For a given output, make a constraint against one, two,
        or three adjacent natural classes. The constraint is a list of
        sets of feature-value tuples. 1/tier_freq of the time, the
        constraint is tier-based."""
        #TODO think about using lexical and tier constraints smartly, not based
        #on frequency in the constraint set. You need a tier constraint when the
        #gw has similar things on a tier and the cw doesn't. You need a lexical constraint
        # when your constraints worked on a really similar word but don't work
        # on this one (harder, because constraint set keeps changing).
        #self.lexical = False
        #if random.randint(1, lexical_freq) == lexical_freq:
            #self.lexical = True
            #self.constraint = random.choice(winners).meaning
        self.gram = random.randint(1, 3)
        self.num_features = num_features
        self.tier = None
        self.use_tier = False
        tiered_winners = []
        # pick tier, then make grams, then find differences. if none, start
        # over? would work but would waste time.
        # make grams, find differences, then make tiers?
        # make grams with and without tiers, find differences, pick one.
        if random.randint(1, tier_freq) == tier_freq and self.gram != 1: # a unigram tier constraint is just a unigram constraint
            self.use_tier = True
            tiered_winners = [self.get_tier(winner) for winner in winners]
            tiered_winners = [winner for winner in tiered_winners if winner != []]
        if len(winners) > 1:
            if len(tiered_winners) > 1:
                winners = tiered_winners
            self.constraint = self.pick_unique_pattern(winners)
        else:
            if tiered_winners != []:
                winners = tiered_winners
            self.constraint = self.pick_any_pattern(winners)

    def get_ngrams(self, word):
        ngrams_in_word = []
        starting_positions = range(len(word) + 1 - self.gram)
        for i in starting_positions:
            ngram = word[i:i + self.gram + 1]
            # tuple segments
            ngram = [tuple(segment) for segment in ngram]
            # tuple ngram
            ngrams_in_word.append(tuple(ngram))
        # get set of ngrams from each word
        return ngrams_in_word

    def pick_unique_pattern(self, winners):
        """Given a list of words, find sequences of segments that are self.gram long
        and, for any adjacent pair of words in the list, exist in only one of the words
        in the pair. Return one such sequence."""
        all_ngrams = []
        different_ngrams = []
        for winner in winners:
            word = winner.sr
            all_ngrams += self.get_ngrams(word)
        all_ngrams = set(all_ngrams)
        #for i in range(len(all_ngrams) - 1): # for pair of winners
            #different_ngrams += (all_ngrams[i] ^ all_ngrams[i + 1])
            #shared_ngrams += (all_ngrams[i] & all_ngrams[i + 1])
        for ngram in all_ngrams:
            occurrences = [winner.sr.count(ngram) for winner in winners]
            if occurrences.count(occurrences[0]) == len(occurrences):
                different_ngrams += ngram
        try:
            pattern = random.choice(different_ngrams)
            for i in range(len(pattern)):
                pattern[i] = random.sample(pattern[i], random.randint(1, self.num_features)) # subset of the features

            pattern = [set(element) for element in pattern]
            assert len(pattern) == self.gram, "constraint length doesn't match self.gram"
            return pattern
        except IndexError: # no different_ngrams (say, unigrams used and only metathesis done)
            return None

    def pick_any_pattern(self, winner):
        """Starting at a random point in the word, find self.gram number of
        segments."""
        word = [segment.iteritems() for segment in winner[0]]
        locus = random.randint(0, len(word) + 1 - self.gram)
        pattern = []
        for gram in range(self.gram):
            segment = list(word[locus])
            natural_class = random.sample(segment, random.randint(1, self.num_features))
            pattern.append(set(natural_class))
            locus += 1
        return pattern

    def get_tier(self, winner): #TODO ask if all kinds of tiers should be searched over
        """Returns a list of the segments in a word that are positive for a certain
        feature. Features currently supported are vowel, consonant, nasal, and strident."""
        tiers = ['vowel', 'consonant', 'nasal', 'strident']
        assert set(tiers) & set(winner.sr[0]) != 0, "feature dictionary doesn't support tiers"
        if self.tier == None: #if creating the constraint, not getting violations
            while True:
                self.tier = random.choice(tiers)
                if self.tier in set(winner.sr[0]):
                    break
        winner_tier = copy.deepcopy(winner)
        winner_tier.sr = [segment for segment in winner.sr if segment[self.tier] == 1]
        return winner_tier

    def get_violation(self, mapping):
        """Finds the number of places in the surface representation
        (including overlapping ones) that match the pattern of the constraint."""
        if self.use_tier == True:
            mapping = self.get_tier(mapping) #FIXME get tier gives a list instead of a mapping, can't get its sr
        violation = 0
        for i in range(len(mapping.sr) + 1 - self.gram):
            segments = mapping.sr[i:i + self.gram]
            assert len(segments) == self.gram, 'slice of sr is the wrong size'
            # if each segment in the slice is in the corresponding natural class
            # of the constraint, add a violation
            #print 'segments', segments
            assert len(segments) == self.gram, "slice length doesn't match self.gram"
            assert len(segments) == len(self.constraint), "slice length doesn't match constraint length"
            matches = map(lambda x, y: set(x.iteritems()) >= y, segments, self.constraint)
            if False not in matches:
                violation += 1
        return violation

class Faithfulness:
    def __init__(self, winners):
        if len(winners) > 0:
            all_changes = [winner.changes for winner in winners]
            different_changes = []
            for i in range(len(all_changes) - 1):
                different_changes.append(set(all_changes[i]) ^ set(all_changes[i + 1]))
            self.constraint = random.choice(different_changes)
        else:
            changes = winners[0].changes
            self.constraint = random.choice(changes)

    def get_violation(self, mapping):
        """Finds the number of times the change referred to by the constraint occurs in the input-output pair."""
        return mapping.changes.count(self.constraint)

class HGGLA:
    def __init__(self, learning_rate, feature_dict, induction, tier_freq):
        """Takes processed input and learns on it one tableau at a time.
        The constraints are updated by the difference in violation vectors
        between the computed winner and the desired winner,
        multiplied by the learning rate."""
        self.learning_rate = learning_rate
        self.constraints = Con(feature_dict, tier_freq)
        self.induction = induction

    def evaluate(self, tableau, if_tie = 'guess'):
        """Use constraints to find mappings violations
        and constraint weights to find mappings harmony scores.
        From harmony scores, find and return the mapping predicted to win."""
        computed_winner = None
        while computed_winner == None:
            harmonies = []
            for mapping in tableau:
                self.constraints.get_violations(mapping)
                #print 'm.violations', mapping.violations
                mapping.harmony = numpy.dot(self.constraints.weights, mapping.violations)
                harmonies.append(mapping.harmony)
            highest_harmony = max(harmonies)
            computed_winners = [mapping for mapping in tableau if mapping.harmony == highest_harmony]
            if len(computed_winners) > 1: # there's a tie
                #print 'computed winners', computed_winners
                if if_tie == 'guess':
                    computed_winner = random.choice(computed_winners)
                else:
                    self.constraints.induce(computed_winners)
            else:
                assert len(computed_winners) == 1, 'no computed winners'
                computed_winner = computed_winners[0]
        assert isinstance(computed_winner, Mapping), 'cw not a mapping'
        return computed_winner

    def train(self, inputs):
        # for iteration in range(10): # learn from the data this many times
        errors = []
        random.shuffle(inputs)
        for tableau in inputs: # learn one tableau at a time
            computed_winner = self.evaluate(tableau, if_tie = 'induce')
            assert isinstance(computed_winner, Mapping), "computed winner isn't a Mapping"
            #print 'c winner', computed_winner
            grammatical_winner = None
            for mapping in tableau:
                #print 'mg', mapping.grammatical
                if mapping.grammatical == True:
                    grammatical_winner = mapping
                    #print 'g winner', grammatical_winner
                    break
            if grammatical_winner != computed_winner:
                assert isinstance(computed_winner, Mapping), "computed winner isn't a Mapping"
                assert isinstance(grammatical_winner, Mapping), "grammatical winner isn't a Mapping"
                difference = grammatical_winner.violations - computed_winner.violations
                self.constraints.weights += difference * self.learning_rate
                self.constraints.induce([computed_winner, grammatical_winner])
                errors.append(1)
            else:
                errors.append(0)
        return errors


    def test(self, inputs):
        winners = [self.evaluate(tableau, if_tie = 'guess') for tableau in inputs]
        return winners

class Learn:
    def __init__(self, feature_chart, input_file_list, algorithm = HGGLA, learning_rate = 0.1, num_trainings = 10, num_negatives = 10, max_changes = 10,
                 processes = '[self.delete, self.metathesize, self.change_feature_value, self.epenthesize]',
                 epenthetics = ['e', '?'], induction = 'comparative', tier_freq = 5):
        feature_dict = FeatureDict(feature_chart)
        inputs = Input(feature_dict, input_file_list[0], num_negatives, max_changes, processes, epenthetics)
        allinput = inputs.allinputs
        test_file = False
        if len(input_file_list) == 2: # there's separate test input
            test_file = True
            test_inputs = Input(feature_dict, input_file_list[1], num_negatives, max_changes, processes, epenthetics)
            testinput = test_inputs.allinputs
        self.alg = algorithm(learning_rate, feature_dict, induction, tier_freq)
        self.num_trainings = num_trainings
        self.accuracy = []
        self.all_errors = []
        self.run_HGGLA(allinput)
        if test_file:
            self.test_HGGLA(testinput)
        print('errors per training', self.all_errors, 'accuracy', self.accuracy,
               'number of constraints', len(self.alg.constraints.constraints))
              #sep = '\n') # file = filename for storing output

    def make_tableaux(self, inputs):
        #print inputs
        urs = [mapping.ur for mapping in inputs if mapping.grammatical == True]
        tableaux = [[mapping for mapping in inputs if (mapping.ur == item).all()] for item in urs]
        return tableaux

    def run_HGGLA(self, inputs):
        tableaux = self.make_tableaux(inputs)
        assert self.alg.constraints.constraints == []
        for i in range(self.num_trainings):
            errors = self.alg.train(tableaux)
            self.all_errors.append(sum(errors))

    def test_HGGLA(self, testinput):
        tableaux = self.make_tableaux(testinput)
        computed_winners = self.alg.test(tableaux)
        for winner in computed_winners:
            if winner.grammatical == True:
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)

class CrossValidate(Learn):
    """Train the algorithm on every possible set of all but one data point
    and test on the leftover data point.
    Look at the average accuracy across tests."""
    #def __init__(self, feature_chart, input_file, algorithm = HGGLA, learning_rate = 0.1, num_negatives = 10, max_changes = 10,
                 #processes = '[self.delete, self.metathesize, self.change_feature_value, self.epenthesize]',
                 #epenthetics = ['e', '?'], induction = 'comparative', tier_freq = 5):
        #feature_dict = FeatureDict(feature_chart)
        #inputs = Input(feature_dict, input_file, num_negatives, max_changes, processes, epenthetics)
        #allinput = inputs.allinputs
        #self.alg = algorithm(learning_rate, feature_dict, induction, tier_freq)
        #if algorithm == HGGLA:
            #self.accuracy = self.run_HGGLA(allinput)

    def refresh_input(self, inputs):
        for tableau in inputs:
            for mapping in tableau:
                mapping.violations = numpy.array([1])
                mapping.harmony = None

    def refresh_con(self):
        self.alg.constraints.constraints = []
        self.alg.constraints.weights = numpy.array([0])

    def run_HGGLA(self, inputs):
        tableaux = self.make_tableaux(inputs)
        for i, tableau in enumerate(tableaux):
            self.refresh_input(tableaux)
            self.refresh_con()
            assert self.alg.constraints.constraints == []
            training_set = tableaux[:i] + tableaux[i + 1:]
            random.shuffle(training_set) #FIXME should I shuffle here and in train?
            for i in range(self.num_trainings):
                errors = self.alg.train(training_set)
                self.all_errors.append(sum(errors))
            # test
            desired = None
            test_tableau = []
            for mapping in tableau:
                if mapping.grammatical == True:
                    desired = mapping.sr
                map_copy = copy.deepcopy(mapping)
                map_copy.grammatical = 'test'
                test_tableau.append(map_copy)
            for m in test_tableau:
                assert m.harmony == None
                assert m.violations == numpy.array([1])
            if self.alg.test([test_tableau])[0].sr == desired: # [0] because it returns a list of one element
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)

#TODO graph all_errors instead of printing
if __name__ == '__main__':
    import os
    import sys
    #localpath = os.getcwd() + '/' + '/'.join(sys.argv[0].split('/')[:-1])
    localpath = '/'.join(sys.argv[0].split('/')[:-1])
    os.chdir(localpath)
    learn1 = Learn('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #learn2 = Learn('feature_chart3.csv', ['input4.csv'], tier_freq = 10)
    #xval1 = CrossValidate('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #xval2 = CrossValidate('feature_chart3.csv', ['input4.csv'], tier_freq = 10)
    #learnTurkish = Learn('TurkishFeaturesSPE.csv', ['TurkishInput1.csv', 'TurkishTest1.csv'], tier_freq = 10)
