#!/usr/bin/env python
import csv, copy, numpy, random, cPickle, datetime
from mapping import Mapping
#THINGS TO WATCH OUT FOR:
    # delete the saved input file if you change the input making code
    # implement tableau making if you give it a file with ungrammatical mappings
from featuredict import FeatureDict
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
#TODO switch from unigram to bigram if no constraints possible
#TODO make non_boundaries dict in FeatureDict and make it read "boundary" from feature names
#TODO make sure there's a faithful mapping?
#TODO Faithfulness: delete a change that's being reversed? make dontcares in major features and make violations notice.

#TODO random to numpy.random - did random.random and randint but not choice or
#sample (which will both be choice, with an optional size arg, have to take an
        #array)
#TODO save output to file; consider tracking SSE
#TODO work on get ngrams (copy problem in diff ngrams), get violations

#TODO data presentation (str method for constraints, graphs for errors), consider setting initial weights positive or negative depending on where they came from
#TODO reward constraints

#Profiler:
    #change feature value is slow
    #array to string is slow. think it's used in getting violations.
    #multiarray.array
    #getviolations, get violation sr, get ngrams, numpy.any
# import cProfile, learner, pstats
# cProfile.run("Learn('feature_chart4.csv', ['input5.csv'], processes =
# '[self.change_feature_value]')", 'learnerprofile.txt')
# p = pstats.Stats('learnerprofile.txt')
# p.sort_stats('cumulative').print_stats(30)
# with input, .414 on input5
# without input, .142 on input5

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
        try:
            saved_file = open('save-' + infile, 'rb')
            self.allinputs = cPickle.load(saved_file)
            print 'read from file'
        except IOError:
            self.allinputs = self.make_input(infile)
            saved_file = open('save-' + infile, 'wb')
            cPickle.dump(self.allinputs, saved_file)
        saved_file.close()
        print 'done making input'

    def make_input(self, infile):
        """Based on file of lines of the form "1,underlyingform,surfaceform,changes"
        create Mapping objects with those attributes.
        Create ungrammatical mappings if not present.
        Bundle mappings into tableaux."""
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
                    mapping.set_ngrams()
                    allinputs.append(mapping)
                    #TODO put into tableaux with dictionary of urs
                else:
                    gen = Gen(self.feature_dict, *self.gen_args)
                    negatives = gen.ungrammaticalize(mapping)
                    mapping.add_boundaries()
                    mapping.set_ngrams()
                    tableau = [mapping] + negatives
                    allinputs.append(tableau)
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
        self.non_boundaries = copy.deepcopy(self.feature_dict.fd)
        del self.non_boundaries['|']

    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly
        ungrammatical mappings with the same input."""
        negatives = []
        while len(negatives) < self.num_negatives:
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), set()])
            for j in range(numpy.random.randint(0, self.max_changes + 1)):
                process = random.choice(self.processes)
                process(new_mapping)
            # Don't add a mapping if it's the same as the grammatical one.
            try:
                if numpy.equal(new_mapping.sr, mapping.sr).all():
                    continue
            except ValueError: # they're not the same length
                pass
            # Don't add a mapping if it's the same as a previously
            # generated one.
            new_mapping.add_boundaries()
            duplicate = False
            for negative in negatives:
                try:
                    if numpy.equal(new_mapping.sr, negative.sr).all():
                        duplicate = True
                        break
                except ValueError:
                    pass
            if duplicate == False:
                new_mapping.set_ngrams()
                negatives.append(new_mapping)
        return negatives

    def epenthesize(self, mapping):
        """Map a ur to an sr with one more segment."""
        epenthetic_segment = random.choice(self.epenthetics)
        epenthetic_features = self.feature_dict.get_features_seg(epenthetic_segment)
        locus = numpy.random.randint(0, len(mapping.sr) + 1)
        mapping.sr = list(mapping.sr)
        mapping.sr.insert(locus, epenthetic_features)
        mapping.sr = numpy.array(mapping.sr)
        mapping.changes.add(frozenset('epenthesize', self.major_features(epenthetic_features)))

    def delete(self, mapping):
        """Map a ur to an sr with one less segment."""
        if len(mapping.sr) > 1:
            locus = numpy.random.randint(0, len(mapping.sr))
            mapping.sr = list(mapping.sr)
            deleted = mapping.sr.pop(locus)
            mapping.sr = numpy.array(mapping.sr)
            mapping.changes.add(frozenset('delete', self.major_features(deleted)))

    def metathesize(self, mapping):
        """Map a ur to an sr with two segments having swapped places."""
        if len(mapping.sr) > 1:
            locus = numpy.random.randint(0, len(mapping.sr) - 1)
            moved_left = mapping.sr[locus + 1]
            mapping.sr = list(mapping.sr)
            moved_right = mapping.sr.pop(locus)
            mapping.sr.insert(locus + 1, moved_right)
            mapping.sr = numpy.array(mapping.sr)
            moved_left_features = [''.join([2, f]) for f in self.major_features(moved_left)]
            mapping.changes.add(frozenset('metathesize', self.major_features(moved_right), moved_left_features))

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
        locus = numpy.random.randint(0, len(mapping.sr))
        segment = mapping.sr[locus]
        major_features = self.major_features(segment)
        closest_num = None
        closest_phone = None
        differences = None
        new_segment = None
        num_to_change = numpy.random.geometric(p = .5, size = 1)
        if num_to_change > self.feature_dict.num_features:
            num_to_change = random.sample(range(1, self.feature_dict.num_features + 1), 1)
        for phone in self.non_boundaries.values():
            differences = segment - phone
            num_different = numpy.count_nonzero(differences)
            if num_different <= num_to_change:
                new_segment = phone
            else:
                if num_different < closest_num or closest_num == None:
                    closest_phone = phone
        if new_segment == None:
            new_segment = closest_phone
        # make a population of numbers representing the number of features to
        # change. the higher the number, the less it is represented. The
        # relationship is linear. Randomly select a number from this population.
        #population = [(-x + self.feature_dict.num_features + 1)*[x] for x in range(1, self.feature_dict.num_features)]
        #population = [x for block in population for x in block]
        # Search the segmental inventory for segments with that many features
        # different from the original one. If none are found, sample again.
        #while new_segment == None:
            #num_changes = random.sample(population, 1)[0]
            #for seg in self.non_boundaries.values():
                #if numpy.count_nonzero(segment - seg) == num_changes:
                    #new_segments.append(seg)
            #try:
                #new_segment = random.choice(new_segments)
            #except IndexError:
                #pass
        #differences = segment - new_segment
        mapping.sr[locus] = new_segment
        changed_features = numpy.nonzero(differences)
        for i in changed_features[0]:
            #new_change = ''.join(['change', str(i), str(new_segment[i])])
            #new_change = major_features + new_change #major features should return a set of keys
            changelist = ['change', str(i), new_segment[i]] + major_features
            mapping.changes.add(frozenset(changelist))

class Con:
    def __init__(self, feature_dict, tier_freq, aligned):
        self.constraints = []
        self.weights = numpy.array([0]) # intercept weight
        self.feature_dict = feature_dict
        self.tier_freq = tier_freq
        self.aligned = aligned
        self.i = 0

    def induce(self, winners):
        """Makes one new markedness and one new faithfulness constraint
        and initializes their weights, unless appropriate constraints
        cannot be found within 15 tries."""
        #print 'calling induce'
        #print self.i
        assert len(self.weights) == len(self.constraints) + 1
        self.make_constraint(Faithfulness, winners, self.feature_dict)
        #print 'made F'
        if self.aligned == True:
            self.make_constraint(MarkednessAligned, self.feature_dict, self.tier_freq, winners)
            #print 'made M'
        else:
            self.make_constraint(Markedness, self.feature_dict, self.tier_freq, winners)
        new_weights = -1 * numpy.random.random(self.num_needed(self.weights)) #for negative weights
        self.weights = numpy.append(self.weights, new_weights)
        assert len(self.weights) == len(self.constraints) + 1
        self.i += 1

    def make_constraint(self, constraint_type, *args):
        i = 0
        while i < 15:
            new_constraint = constraint_type(*args)
            i += 1
            if new_constraint.constraint == None:
                break
            elif new_constraint not in self.constraints:
                self.constraints.append(new_constraint)
                break

    def get_violations(self, mapping):
        #print 'evaluating'
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
        self.constraint = None
        self.gram = numpy.random.randint(1, 4) #FIXME avoid setting too high a gram for the word
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
        if numpy.random.randint(1, self.tier_freq + 1) == self.tier_freq:
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
        self.dontcares = [random.sample(range(self.num_features), numpy.random.randint(0, self.num_features)) for segment in ngram]
        self.cares = []
        for segment in self.dontcares:
            for feature in segment:
                ngram[segment][feature] = 0

    def make_care(self, pattern):
        segment = numpy.random.randint(0, self.gram)
        feature = self.dontcares[segment].pop(random.choice(range(len(segment))))
        self.constraint[segment][feature] = pattern[segment][feature]

    def distinguishes_winners(self, pattern, winners):
        violations = [self.get_violation(winner) for winner in winners]
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
        start = numpy.random.randint(0, len(winner) + 2 - self.gram)
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

    def get_violation(self, mapping):
        """Finds the number of places in the surface representation
        (including overlapping ones) that match the pattern of the constraint."""
        violation = 0
        ngrams = mapping.ngrams[self.gram - 1] if self.tier == None else mapping.get_ngrams(self.get_tier(mapping.sr), self.gram)
        #print len(ngrams) # 9989 10 10 10
        for ngram in ngrams:
            if (numpy.absolute(ngram - self.constraint) > 1).any() == False:
                violation += 1
        return violation

class MarkednessAligned(Markedness):
    def pick_unique_pattern(self, winners):
        """Chooses two winners if there are more than two. Finds all differences
        between them and picks one to preserve in the constraint. Bases the
        constraint off of one of the two winners, but makes features don't-cares
        at random. Does not allow the protected feature to become a
        don't-care."""
        winner = winners[1] #for negative constraints; could randomly choose for a mix of negative and positive
        diff_array = winners[0] - winners[1]
        differences = numpy.nonzero(numpy.absolute(diff_array) > 1) # indices of differences
        #print winners[0], winners[1], differences
        assert differences[0].size > 0, 'duplicates'
        ind = random.choice(range(len(differences[0])))
        segment = differences[0][ind]
        feature = differences[1][ind]
        positions = range(self.gram)
        random.shuffle(positions)
        for position_in_ngram in positions:
            pattern = copy.copy(winner[segment - position_in_ngram:segment + self.gram - position_in_ngram])
            if len(pattern) == self.gram:
                self.constraint = pattern
                break
        #print self.constraint
        indices = numpy.where(numpy.random.random(self.constraint.shape) > numpy.random.random(1))
        #print self.constraint.shape, segment, feature
        saved_constraint = self.constraint[position_in_ngram][feature]
        self.constraint[indices] = 0
        self.constraint[position_in_ngram][feature] = saved_constraint
        #assert (winners[0]).all(), 'dontcares affected winner'

    def __str__(self):
        segments = []
        for segment in self.constraint:
            natural_class = [k for k, v in self.feature_dict.fd.iteritems() if (numpy.absolute(segment - v) < 2).all()]
            segments.append(str(natural_class))
        if self.tier != None:
            return ' '.join([self.feature_dict.feature_names[self.tier], 'tier', str(segments)])
        else:
            return str(segments)

class Faithfulness:
    def __init__(self, winners, feature_dict):
        """Find a change that exists in only one winner. Abstract away from some
        of its feature values, but not so much that it becomes equivalent to a
        change in the other winner. Make this a faithfulness constraint."""
        self.feature_dict = feature_dict
        try:
            self.constraint = set((winners[1].changes - winners[0].changes).pop())
            if 'metathesis' in self.constraint:
                dontcares = random.sample(['consonant:1', 'vowel:1', 'sonorant:1', '2consonant:1', '2vowel:1', '2sonorant:1', 'consonant:-1', 'vowel:-1', 'sonorant:-1', '2consonant:-1', '2vowel:-1', '2sonorant:-1'], numpy.random.randint(0, 13))
            else:
                dontcares = random.sample(['consonant:1', 'vowel:1', 'sonorant:1', 'consonant:-1', 'vowel:-1', 'sonorant:-1'], numpy.random.randint(0, 7))
            dontcares = set(dontcares) & self.constraint # so if you add things back, you don't add something that wasn't there to begin with
            self.constraint -= dontcares
            for change in winners[0].changes:
                while self.constraint <= change:
                    docare = dontcares.pop()
                    self.constraint.add(docare)
        except KeyError:
            self.constraint = None

    def get_violation(self, mapping):
        """Finds the number of times the change referred to by the constraint occurs in the input-output pair."""
        violation = 0
        for change in mapping.changes:
            if self.constraint <= change:
                violation += 1
        return violation

    def __str__(self):
        segment_type = []
        new_value = None
        feature = None
        process_type = None
        for item in self.constraint:
            if type(item) == numpy.int32 or int:
                new_value = item
            elif ':' in item:
                segment_type.append(item)
            elif len(item) < 3:
                feature = self.feature_dict.feature_names[item]
            else:
                process_type = item
        segment_type.sort()
        if feature != None:
            return ' '.join([process_type, feature, 'to', new_value, 'in', str(segment_type)])
        else:
            return ' '.join([process_type, str(segment_type)])

class HGGLA:
    def __init__(self, learning_rate, feature_dict, aligned, tier_freq):
        """Takes processed input and learns on it one tableau at a time.
        The constraints are updated by the difference in violation vectors
        between the computed winner and the desired winner,
        multiplied by the learning rate."""
        self.learning_rate = learning_rate
        self.con = Con(feature_dict, tier_freq, aligned)

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
            if mapping.grammatical == True:
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
        errors = []
        random.shuffle(inputs)
        for tableau in inputs: # learn one tableau at a time
            (grammatical_winner, computed_winner, correct) = self.evaluate(tableau)
            if correct:
                if len(computed_winner) == 1:
                    errors.append(0)
                else:
                    computed_winner.remove(grammatical_winner)
                    self.con.induce([grammatical_winner, computed_winner[0]])
                    errors.append(1)
            else:
                if numpy.random.randint(0, 10) == 9:
                    self.con.induce([grammatical_winner, computed_winner[0]])
                else:
                    self.update(grammatical_winner, computed_winner[0])
                errors.append(1)
        return errors

    def test(self, tableau):
        computed_winner = None
        harmonies = []
        for mapping in tableau: # iteration over nonsequence when testing
            self.con.get_violations(mapping)
            mapping.harmony = numpy.dot(self.con.weights, mapping.violations)
            harmonies.append(mapping.harmony)
            highest_harmony = max(harmonies)
            computed_winner = [mapping for mapping in tableau if mapping.harmony == highest_harmony]
        computed_winner = computed_winner[0] if len(computed_winner) == 1 else random.choice(computed_winner)
        return computed_winner

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
        self.output = 'Output-' + str(datetime.datetime.now())
        self.run_HGGLA(allinput)
        if test_file:
            self.test_HGGLA(testinput)
        #for c in self.alg.con.constraints:
            #print c
        with open(self.output, 'a') as f:
            f.write('\n'.join(['\n\nmean testing accuracy',
                               str(numpy.mean(self.accuracy)),
                               'number of constraints',
                               str(len(self.alg.con.constraints)),
                               'constraints'#,
                               #'\n'.join([str(c) for c in self.alg.con.constraints])
                              ]))
            print('errors per training', self.all_errors, 'mean accuracy on test', numpy.mean(self.accuracy),
               'number of constraints', len(self.alg.con.constraints))
              #sep = '\n') # file = filename for storing output

    def run_HGGLA(self, inputs):
        assert self.alg.con.constraints == []
        for i in range(self.num_trainings):
            errors = self.alg.train(inputs)
            sum_errors = sum(errors)
            self.all_errors.append(sum_errors)
            with open(self.output, 'a') as f:
                f.write(''.join(['\nsum of errors for training #', str(i), ': ', str(sum_errors)]))
                f.write(''.join(['number of constraints for training #', str(i), ': ', str(len(self.alg.con.constraints))]))
                if sum_errors != 0:
                    f.write('\nerrors: ' + str(errors))
                f.write('\nall errors: ' + str(self.all_errors))

    def test_HGGLA(self, testinput):
        for tableau in testinput:
            winner = self.alg.test(tableau)
            if winner.grammatical:
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
        self.alg.con.constraints = []
        self.alg.con.constraints.weights = numpy.array([0])

    def run_HGGLA(self, inputs):
        tableaux = self.make_tableaux(inputs)
        for i, tableau in enumerate(tableaux):
            self.refresh_input(tableaux)
            self.refresh_con()
            assert self.alg.con.constraints == []
            training_set = tableaux[:i] + tableaux[i + 1:]
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
    #learn1 = Learn('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #learn2 = Learn('feature_chart4.csv', ['input5.csv'], processes = '[self.change_feature_value]', max_changes = 5, num_negatives = 5, tier_freq = 10)
    #xval1 = CrossValidate('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #xval2 = CrossValidate('feature_chart3.csv', ['input4.csv'], tier_freq = 10)
    learnTurkish = Learn('TurkishFeaturesWithNA.csv', ['TurkishInput2.csv', 'TurkishTest2.csv']
                         , num_trainings = 3, max_changes = 5, num_negatives = 5, tier_freq = 10, processes = '[self.change_feature_value]')
    #TurkishInput2 has the ~ inputs taken out, the variable inputs taken out, and deletion taken out.
    #TurkishInput1 is the same but deletion is still in.
    #same pattern for test files
