#!/usr/bin/env python
import csv, copy, random, numpy, itertools
#TODO get numpy working with python3
#TODO implicational constraints
#TODO consider constraining GEN
#TODO graphs - error rate
#TODO think about using lexical and tier constraints smartly, not based
        #on frequency in the constraint set. You need a tier constraint when the
        #gw has similar things on a tier and the cw doesn't. You need a lexical constraint
        # when your constraints worked on a really similar word but don't work
        # on this one (harder, because constraint set keeps changing).
#TODO consider adding lexically conditioned constraints
#TODO take deletions out of turkish input while i'm not doing deletion
#TODO save input

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
        print 'calling Input'
        self.feature_dict = feature_dict
        self.gen_args = [num_negatives, max_changes, processes, epenthetics]
        # if there's a file with a name based on infile, use it. else:
        self.allinputs = self.make_input(infile)

    def make_input(self, infile):
        """Based on file of lines of the form "1,underlyingform,surfaceform,changes"
        create Mapping objects with those attributes.
        Make words into lists of segments,
        segments into dictionaries of feature values,
        and changes into a list of strings.
        Create ungrammatical mappings if not present."""
        allinputs = []
        ungrammatical_included = False
        with open(infile, 'r') as f:
            fread = list(csv.reader(f))
            if fread[1][0] == '0':
                ungrammatical_included = True
            for line in fread:
                mapping = Mapping(self.feature_dict, line)
                mapping.to_data()
                if ungrammatical_included:
                    mapping.add_boundaries()
                    allinputs.append(mapping)
                else:
                    gen = Gen(self.feature_dict, *self.gen_args)
                    negatives = gen.ungrammaticalize(mapping)
                    mapping.add_boundaries()
                    allinputs.append(mapping)
                    allinputs += negatives
            return allinputs

class Gen:
    """Generates input-output mappings."""
    def __init__(self, feature_dict, num_negatives, max_changes, processes, epenthetics):
        self.feature_dict = feature_dict
        self.major_features = self.feature_dict.major_features
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = eval(processes)
        self.epenthetics = epenthetics

    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly ungrammatical mappings with the same input."""
        negatives = []
        while len(negatives) < self.num_negatives:
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
            for j in range(random.randint(0, self.max_changes)):
                process = random.choice(self.processes)
                process(new_mapping)
            # Don't add a mapping if it's the same as the grammatical one.
            try:
                if numpy.equal(new_mapping.sr, mapping.sr).all():
                    #print 'process didnt apply'
                    continue
            except ValueError: # they're not the same length
                pass
            # Don't add a mapping if it's the same as a previously
            # generated one.
            duplicate = False
            for negative in negatives:
                try:
                    if numpy.equal(new_mapping.sr, negative.sr).all():
                        duplicate = True
                        #print 'made duplicate'
                        break
                except ValueError:
                    pass
            if duplicate == False:
                new_mapping.add_boundaries()
                negatives.append(new_mapping)
                #print 'appended'
        return negatives

    def epenthesize(self, mapping):
        """Map a ur to an sr with one more segment."""
        epenthetic_segment = random.choice(self.epenthetics)
        epenthetic_features = self.feature_dict.get_features_seg(epenthetic_segment)
        locus = random.randint(0, len(mapping.sr))
        mapping.sr = list(mapping.sr)
        mapping.sr.insert(locus, epenthetic_features)
        mapping.sr = numpy.array(mapping.sr)
        new_change = ['epenthesize', self.major_features(epenthetic_features)]
        mapping.changes.append(' '.join([`item` for item in new_change]))

    def delete(self, mapping):
        """Map a ur to an sr with one less segment."""
        if len(mapping.sr) > 1:
            locus = random.randint(0, len(mapping.sr) - 1)
            mapping.sr = list(mapping.sr)
            deleted = mapping.sr.pop(locus)
            mapping.sr = numpy.array(mapping.sr)
            new_change = ['delete', self.major_features(deleted)]
            mapping.changes.append(' '.join([`item` for item in new_change]))

    def metathesize(self, mapping):
        """Map a ur to an sr with two segments having swapped places."""
        if len(mapping.sr) > 1:
            locus = random.randint(0, len(mapping.sr) - 2)
            moved_left = mapping.sr[locus + 1]
            mapping.sr = list(mapping.sr)
            moved_right = mapping.sr.pop(locus)
            mapping.sr.insert(locus + 1, moved_right)
            mapping.sr = numpy.array(mapping.sr)
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
        """Map a ur to an sr with one segment changed. Prefers fewer feature
        changes, but will only map to segments that are in the inventory. Each
        feature change is recorded in rule format: 'change
        feature-index new-feature-value
        major-features-of-original-segment'."""
        locus = random.randint(0,
        len(mapping.sr) - 1)
        segment = mapping.sr[locus]
        major_features = self.major_features(segment)
        new_segments = []
        new_segment = None
        # make a population of numbers representing the number of features to
        # change. the higher the number, the less it is represented. The
        # relationship is linear. Randomly select a number from this population.
        population = [(-x + self.feature_dict.num_features + 1)*[x] for x in range(self.feature_dict.num_features)]
        population = [x for block in population for x in block]
        # Search the segmental inventory for segments with that many features
        # different from the original one. If none are found, sample again.
        while new_segment == None:
            num_changes = random.sample(population, 1)[0]
            for seg in self.feature_dict.fd.values():
                if numpy.sum(numpy.absolute(segment - seg)) == 2 * num_changes:
                    new_segments.append(seg)
            try:
                new_segment = random.choice(new_segments)
            except IndexError:
                pass
        mapping.sr[locus] = new_segment
        differences = segment - new_segment
        changed_features = numpy.nonzero(differences)
        for i in changed_features:
            mapping.changes.append(' '.join(['change', str(i), str(new_segment[i]), str(major_features)]))

class Mapping:
    def __init__(self, feature_dict, line):
        """Each input-output mapping is an object with attributes: grammatical (is it grammatical?),
        ur (underlying form), sr (surface form), changes (operations to get from ur to sr),
        violations (of constraints in order), and harmony (violations times constraint weights).
        ur and sr are numpy arrays, with segments as rows and features as columns."""
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
            changeparts = self.changes[i].split(' ')
            for j in range(len(changeparts)):
                if len(changeparts[j]) == 1 and changeparts[j] not in ['0', '1']: #segment
                    changeparts[j] = str(self.feature_dict.major_features(self.feature_dict.get_features_seg(changeparts[j])))
            if changeparts[0] == 'change':
                changeparts[1] = str(self.feature_dict.feature_names.index(changeparts[1]))
            self.changes[i] = ' '.join(changeparts)

    def add_boundaries(self):
        boundary = self.feature_dict.fd['|']
        self.ur = numpy.vstack((boundary, self.ur, boundary))
        self.sr = numpy.vstack((boundary, self.sr, boundary))

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
                segment = line.pop(0)
                if i == 0:
                    self.feature_names = line
                if i > 0:
                    self.num_features = len(line)
                    for j in range(len(line)):
                        line[j] = int(line[j])
                        line[j] = -1 if line[j] == 0 else line[j]
                    self.fd[segment] = numpy.array(line)
        self.tiers = self.init_tiers()

    def get_features_seg(self, segment):
        return copy.copy(self.fd[segment])

    def get_features_word(self, word):
        features = [self.get_features_seg(segment) for segment in word]
        features = numpy.array(features)
        return features

    def get_segments(self, word):
        return [self.get_segment(features) for features in word]

    def get_segment(self, features):
        return [k for k, v in self.fd.iteritems() if numpy.equal(v, features).all()][0]

    def major_features(self, featureset):
        """Select only the first three features. These should be vocalic, consonantal, and sonorant."""
        return featureset[0:3]

    def init_tiers(self):
        tiers = []
        tier_names = ['vocalic', 'consonantal', 'nasal', 'strident']
        for name in tier_names:
            try:
                tiers.append(self.feature_names.index(name))
            except ValueError:
                pass
        assert len(tiers) > 0, "feature chart doesn't support tiers"
        return tiers

class Con:
    def __init__(self, feature_dict, tier_freq):
        self.constraints = []
        self.weights = numpy.array([0]) # intercept weight
        self.feature_dict = feature_dict
        self.tier_freq = tier_freq

    def induce(self, winners, aligned):
        """Makes one new markedness and one new faithfulness constraint
        and initializes their weights, unless appropriate constraints
        cannot be found within 15 tries."""
        print 'calling induce'
        assert len(self.weights) == len(self.constraints) + 1
        if aligned == True:
            self.make_constraint(MarkednessAligned, self.feature_dict, self.tier_freq, winners)
        else:
            self.make_constraint(Markedness, self.feature_dict, self.tier_freq, winners)
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
        print 'evaluating'
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
    def __init__(self, feature_dict, tier_freq, winners):
        """For a given output, make a constraint against one, two,
        or three adjacent natural classes. 1/tier_freq of the time, the
        constraint is tier-based."""
        #print 'markedness init'
        winners = [winner.sr for winner in winners]
        self.constraint = []
        self.gram = random.randint(1, 3) #FIXME avoid setting too high a gram for the word
        self.feature_dict = feature_dict
        self.num_features = self.feature_dict.num_features
        self.tier = None
        self.tier_freq = tier_freq
        tier_winners = []
        if self.gram != 1: # a unigram tier constraint is just a unigram constraint
            winners = self.decide_tier(winners)
        if len(winners) > 1:
            self.pick_unique_pattern(winners)
        else:
            self.pick_any_pattern(winners)

    def decide_tier(self, winners):
        """Randomly decide whether to have a tier constraint. If yes, call
        get_tiers on winners. Remove any winners that have none of the chosen
        tier. If there are fewer than the desired number of winners (one or
        at least two), then decide not to use a tier after all."""
        if random.randint(1,
        self.tier_freq) == self.tier_freq:
            tier_winners = [self.get_tier(winner) for winner in winners]
            tier_winners = [winner for winner in winners if winner != []]
            desired_number = 1 if len(winners) == 1 else 2
            if len(tier_winners) >= desired_number:
                winners = tier_winners
            else:
                self.tier = None
        return winners

    def get_ngrams(self, word):
        starting_positions = range(len(word) + 1 - self.gram)
        ngrams_in_word = numpy.array([word[i:i + self.gram] for i in starting_positions])
        return ngrams_in_word

    def different_ngram(self, all_ngrams):
        """Finds the set of all ngrams in the winners, and finds one ngram that
        does not occur the same number of times in all winners."""
        list_all_ngrams = all_ngrams.tolist()
        flattened_ngrams = [ngram for word in list_all_ngrams for ngram in word]
        sorted_ngrams = flattened_ngrams.sort()
        unique_ngrams = [n for i, n in enumerate(sorted_ngrams) if i == 0 or n != sorted_ngrams[i-1]]
        for ngram in random.shuffle(unique_ngrams):
            occurrences = [word.count(ngram) for word in all_ngrams]
            if occurrences.count(occurrences[0]) != len(occurrences):
                return numpy.asarray(ngram)

    def dont_care(self, ngram):
        self.dontcares = [random.sample(range(self.num_features), random.randint(0, self.num_features - 1)) for segment in ngram]
        self.cares = []
        for segment in self.dontcares:
            for feature in segment:
                ngram[segment][feature] = 0

    def make_care(self, pattern):
        segment = random.randint(0, self.gram - 1)
        feature = self.dontcares[segment].pop(random.choice(range(len(segment))))
        self.constraint[segment][feature] = pattern[segment][feature]

    def distinguishes_winners(self, pattern, winners):
        violations = [self.get_violation_sr(winner) for winner in winners]
        return violations.count(violations[0]) != len(violations)

    def pick_unique_pattern(self, winners):
        """Choose a constraint from a list of words. The constraint should be
        self.gram segments long and capable of distinguishing between two of the
        words in the list."""
        all_ngrams = numpy.array([self.get_ngrams(winner) for winner in winners])
        self.constraint = self.different_ngram(all_ngrams)
        if self.constraint != None:
            pattern = self.constraint
            self.dont_care(self.constraint)
            assert len(self.constraint) == self.gram, "constraint length doesn't match self.gram"
            while self.distinguishes_winners(self.constraint, winners) == False:
                self.make_care(pattern)

    def pick_any_pattern(self, winner):
        """Starting at a random point in the word, find self.gram number of
        segments."""
        start = random.randint(0, len(winner) + 1 - self.gram)
        ngram = winner[start:start + self.gram]
        ngram = self.dontcares(ngram)
        self.constraint = ngram

    def get_tier(self, winner): #TODO ask if all kinds of tiers should be searched over
        """Returns a list of the segments in a word that are positive for a certain
        feature. Features currently supported are vocalic, consonantal, nasal, and strident."""
        if self.tier == None: #if creating the constraint, not getting violations
            self.tier = random.choice(self.feature_dict.tiers)
        winner_tier = [segment for segment in winner if segment[self.tier] == 1]
        return winner_tier

    def get_violation_sr(self, sr):
        violation = 0
        ngrams = self.get_ngrams(sr)
        for ngram in ngrams:
            if (numpy.absolute(ngram - self.constraint) > 1).any() == False:
                violation += 1
        return violation

    def get_violation(self, mapping):
        """Finds the number of places in the surface representation
        (including overlapping ones) that match the pattern of the constraint."""
        winner = mapping.sr
        if self.tier != None:
            winner = self.get_tier(winner)
        return self.get_violation_sr(winner)

class MarkednessAligned(Markedness):
    def pick_unique_pattern(self, winners):
        """Chooses two winners if there are more than two. Finds all differences
        between them and picks one to preserve in the constraint. Bases the
        constraint off of one of the two winners, but makes features don't-cares
        at random. Does not allow the protected feature to become a
        don't-care."""
        winners = random.sample(winners, 2)
        winner = winners[0]
        diff_array = winners[0] - winners[1]
        differences = numpy.nonzero(diff_array) # indices of differences
        found_ngram = False
        while self.constraint == []:
            ind = random.choice(range(len(differences[0])))
            segment = differences[0][ind]
            feature = differences[1][ind]
            position_in_ngram = random.choice(range(self.gram))
            pattern = winner[segment - position_in_ngram:segment + self.gram - position_in_ngram]
            if len(pattern) == self.gram:
                self.constraint = pattern
        indices = [random.sample(range(self.num_features), random.randint(0, self.num_features -1)) for seg in range(len(self.constraint))]
        for seg in range(len(indices)):
            for feat in indices[seg]:
                if ((seg == segment) and (feat == feature)) == False:
                    self.constraint[seg][feat] = 0

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
    def __init__(self, learning_rate, feature_dict, aligned, tier_freq):
        """Takes processed input and learns on it one tableau at a time.
        The constraints are updated by the difference in violation vectors
        between the computed winner and the desired winner,
        multiplied by the learning rate."""
        self.learning_rate = learning_rate
        self.constraints = Con(feature_dict, tier_freq)
        self.aligned = aligned

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
                    self.constraints.induce(computed_winners, self.aligned)
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
                self.constraints.induce([computed_winner, grammatical_winner], self.aligned)
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
                 epenthetics = ['e', '?'], aligned = True, tier_freq = 5):
        feature_dict = FeatureDict(feature_chart)
        inputs = Input(feature_dict, input_file_list[0], num_negatives, max_changes, processes, epenthetics)
        allinput = inputs.allinputs
        test_file = False
        if len(input_file_list) == 2: # there's separate test input
            test_file = True
            test_inputs = Input(feature_dict, input_file_list[1], num_negatives, max_changes, processes, epenthetics)
            testinput = test_inputs.allinputs
        self.alg = algorithm(learning_rate, feature_dict, aligned, tier_freq)
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
        tableaux = [[mapping for mapping in inputs if (mapping.ur.shape == item.shape and numpy.equal(mapping.ur, item).all())] for item in urs]
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
            random.shuffle(training_set)
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
            if self.alg.test([test_tableau])[0].sr == desired: # returns a list of one element
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)

if __name__ == '__main__':
    import os
    import sys
    #localpath = os.getcwd() + '/' + '/'.join(sys.argv[0].split('/')[:-1])
    localpath = '/'.join(sys.argv[0].split('/')[:-1])
    os.chdir(localpath)
    #learn1 = Learn('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #learn2 = Learn('feature_chart3.csv', ['input4.csv'], processes = '[self.change_feature_value]', max_changes = 5, num_negatives = 5, tier_freq = 10)
    #xval1 = CrossValidate('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #xval2 = CrossValidate('feature_chart3.csv', ['input4.csv'], tier_freq = 10)
    learnTurkish = Learn('TurkishFeaturesSPE.csv', ['TurkishInput1.csv', 'TurkishTest1.csv'], tier_freq = 10, processes = '[self.change_feature_value]')
