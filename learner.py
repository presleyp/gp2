#!/usr/bin/env python
import csv, copy, numpy, random, cPickle, datetime
import matplotlib.pyplot as pyplot
from mapping import Mapping
from featuredict import FeatureDict
from matplotlib.backends.backend_pdf import PdfPages
#THINGS TO WATCH OUT FOR:
    # delete the saved input file if you change the input making code
    # implement tableau making if you give it a file with ungrammatical mappings
#TODO make non_boundaries dict in FeatureDict and make it read "boundary" from feature names
#TODO Faithfulness: delete a change that's being reversed? make sure there's a faithful mapping?
#TODO time random.sample vs numpy.random.sample
#TODO copy problem in diff_ngrams
# keep in mind: tiers can mess up alignment

#TODO think about using losers to help guide constraint induction. if you make a constraint, you want it to privilege cw over gw, but you
# also want cw to win over losers. could this help?
#TODO look at Jason Riggle's GEN and think about using CON to make GEN.

#TODO move classes into different files? extract polarity functions and get ngrams?
#TODO add parameters, continue fixing output

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
        #self.major_features = self.feature_dict.major_features
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
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
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
            if not duplicate:
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
        #major_features = self.major_features(segment)
        closest_num = None
        closest_phone = None
        differences = None
        new_segment = None
        #num_to_change = numpy.random.geometric(p = .5, size = 1)[0]
        num_to_change = numpy.random.zipf(2)
        if num_to_change > len(segment):
            num_to_change = random.sample(range(1, len(segment) + 1), 1)[0]
        for phone in self.non_boundaries.values():
            differences = segment - phone # FIXME not sure this makes sense
            num_different = len(differences)
            if num_different == num_to_change:
                new_segment = phone
            else:
                if num_different != 0 and (closest_num == None or numpy.absolute(num_different - num_to_change) < numpy.absolute(closest_num - num_to_change)):
                    closest_num = num_different
                    closest_phone = phone
        if not new_segment:
            new_segment = closest_phone
        mapping.sr[locus] = new_segment
        changed_features = segment - new_segment
        assert changed_features, 'no change made'
        #TODO major features as tuple or frozen set?
        mapping.changes = [set(['change'] + mapping.split(feature)) for feature in changed_features]

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
        #print self.i
        assert len(self.weights) == len(self.constraints) + 1
        if random.random() < .5:
            self.make_constraint(Faithfulness, winners, self.feature_dict)
        else:
            if self.aligned:
                self.make_constraint(MarkednessAligned, self.feature_dict, self.tier_freq, winners)
            else:
                self.make_constraint(Markedness, self.feature_dict, self.tier_freq, winners)
        assert len(self.weights) == len(self.constraints) + 1
        self.i += 1

    def make_constraint(self, constraint_type, *args):
        i = 0
        while i < 10:
            new_constraint = constraint_type(*args)
            i += 1
            if new_constraint.constraint == None:
                break
            duplicate = False
            for constraint in self.constraints:
                if new_constraint == constraint:
                    duplicate = True
                    break
            if not duplicate:
                self.constraints.append(new_constraint)
                self.weights = numpy.append(self.weights, numpy.random.random(1))
                break

    def get_violations(self, mapping):
        new_constraints = -(len(self.constraints) - (len(mapping.violations) - 1))
        if new_constraints < 0:
            new_violations = numpy.array([constraint.get_violation(mapping)
                                          for constraint in self.constraints[new_constraints:]])
            mapping.violations = numpy.append(mapping.violations, new_violations)
        assert len(mapping.violations) == len(self.constraints) + 1

class Markedness:
    def __init__(self, feature_dict, tier_freq, winners):
        """For a given output, make a constraint against one, two,
        or three adjacent natural classes. tier_freq of the time, the
        constraint is tier-based."""
        winners = [winner.sr for winner in winners]
        self.constraint = None
        self.feature_dict = feature_dict
        self.num_features = self.feature_dict.num_features
        self.tier_freq = tier_freq
        self.tier = None
        winners = self.decide_tier(winners)
        self.pick_gram(winners)
        if len(winners) > 1:
            self.pick_unique_pattern(winners)
        else:
            self.pick_any_pattern(winners)

    def pick_gram(self, winners):
        lengths = [len(winner) for winner in winners] + [3]
        ceiling = min(lengths)
        self.gram = numpy.random.randint(1, ceiling + 1)

    def decide_tier(self, winners):
        """Randomly decide whether to have a tier constraint. If yes, call
        get_tiers on winners. Remove any winners that have none of the chosen
        tier. If there are fewer than the desired number of winners (one or
        at least two), then decide not to use a tier after all. If the winners
        are not different, decide not to use a tier. If the
        winners have different lengths, decide not to use a tier.""" #change that when not using MarkednessAligned anymore
        if numpy.random.random() <= self.tier_freq:
            tier_winners = [self.get_tier(winner) for winner in winners]
            tier_winners = [winner for winner in tier_winners if len(winner) != 0]
            #tier_winners = [winner for winner in tier_winners if winner != []]
            desired_number = 1 if len(winners) == 1 else 2
            try:
                if len(tier_winners) >= desired_number and (tier_winners[0] - tier_winners[1]).any():
                    winners = tier_winners
                else:
                    self.tier = None
            except ValueError: #the winners are not the same length
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
        if self.constraint:
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
        if not self.tier: #if creating the constraint, not getting violations
            self.tier = random.sample(self.feature_dict.tiers, 1)[0]
        winner_tier = numpy.array([segment for segment in winner if self.tier in segment])
        return winner_tier

    def get_violation(self, mapping):
        """Finds the number of places in the surface representation
        (including overlapping ones) that match the pattern of the constraint."""
        violation = 0
        ngrams = mapping.ngrams[self.gram - 1] if self.tier == None else mapping.get_ngrams(self.get_tier(mapping.sr), self.gram)
        #print len(ngrams) # 9989 10 10 10
        for ngram in ngrams:
            if (self.constraint <= ngram).all():
                violation += self.violation
        return violation

class MarkednessAligned(Markedness):
    def pick_unique_pattern(self, winners):
        """Chooses two winners if there are more than two. Finds all differences
        between them and picks one to preserve in the constraint. Bases the
        constraint off of one of the two winners, but makes features don't-cares
        at random. Does not allow the protected feature to become a
        don't-care."""
        (base, difference) = self.coinflip(winners)
        assert difference.any(), 'duplicates'
        protected_segment = random.choice(numpy.where(difference)[0])
        protected_feature = random.sample(difference[protected_segment], 1)[0]
        # pick ngram
        positions = range(self.gram)
        random.shuffle(positions)
        position = None
        for position_in_ngram in positions:
            pattern = copy.copy(base[protected_segment - position_in_ngram:protected_segment + self.gram - position_in_ngram])
            if len(pattern) == self.gram:
                self.constraint = pattern
                position = position_in_ngram
                break
        assert self.constraint != None
        for i in range(len(pattern)):
            length = len(pattern[i])
            if length > 1:
                num_features = numpy.random.zipf(2)
                if num_features > length:
                    num_features = numpy.random.randint(1, length)
                pattern[i] = set(random.sample(pattern[i], num_features))
        assert type(pattern[position]) == set
        pattern[position].add(protected_feature)
        self.constraint = pattern

    def coinflip(self, winners):
        """Flip a coin. If heads, the constraint will be made from the grammatical winner and will assign rewards. If tails,
        it will be made from the computed winner and will assign violations. Return the winner the constraint will be made from,
        and the difference between it and the other winner."""
        if numpy.random.randint(0,2) == 1:
            self.violation = 1
            return (winners[0], winners[0] - winners[1])
        else:
            self.violation = -1
            return (winners[1], winners[1] - winners[0])

    def polarity(self, input):
        if input < 0:
            return '-'
        else:
            return '+'

    def __eq__(self, other):
        if len(self.constraint) != len(other.constraint):
            return False
        else:
            return (numpy.equal(self.constraint, other.constraint)).all()

    def __str__(self):
        polarity = '+' if self.violation else '-'
        segments = []
        for segment in self.constraint:
            #natural_class = [k for k, v in self.feature_dict.fd.iteritems() if segment <= v]
            natural_class = [self.polarity(feature) + self.feature_dict.feature_names[numpy.absolute(feature)] for feature in segment]
            natural_class = ''.join(['{', ','.join(natural_class), '}'])
            segments.append(natural_class)
        if self.tier:
            return ''.join([self.feature_dict.feature_names[self.tier], ' tier ', polarity, str(segments)])
        else:
            return ''.join([self.polarity(self.violation)] + [segment for segment in segments])

class Faithfulness:
    def __init__(self, winners, feature_dict):
        """Find a change that exists in only one winner. Abstract away from some
        of its feature values, but not so much that it becomes equivalent to a
        change in the other winner. Make this a faithfulness constraint."""
        self.feature_dict = feature_dict
        self.constraint = None
        base = copy.copy(winners[1].changes)
        assert type(base) == list
        random.shuffle(base)
        for change in base:
            if base.count(change) > winners[0].changes.count(change):
                self.constraint = change
                if numpy.random.random() > .5:
                    polarity = self.constraint & set(['+', '-'])
                    self.constraint -= polarity
                    v_base = 0
                    v_other = 0
                    for change in base:
                        if self.constraint <= change:
                            v_base += 1
                    for change in winners[0].changes:
                        if self.constraint <= change:
                            v_other += 1
                    if v_base <= v_other:
                        self.constraint |= polarity
                break

    def get_violation(self, mapping):
        """Finds the number of times the change referred to by the constraint occurs in the input-output pair."""
        violation = 0
        for change in mapping.changes:
            if self.constraint <= change:
                violation -= 1
        return violation

    def __eq__(self, other):
        self.constraint == other.constraint

    def __str__(self):
        #segment_type = []
        value = None
        feature = None
        process_type = None
        for item in self.constraint:
            if type(item) == numpy.int32:
                feature = self.feature_dict.feature_names[-item]
            elif item in ('-', '+'):
                value = item
            else:
                process_type = item
                if process_type == 'change':
                    process_type = 'Ident'
        if type(process_type) != str:
            print 'process type', process_type, type(process_type)
        assert type(process_type) == str, 'change not str'
        #segment_type.sort()
        if feature:
            if value:
                return ''.join([process_type, ' ', value, feature]) #, 'in', str(segment_type)])
            else:
                return ' '.join([process_type, feature]) #, 'in', str(segment_type)])
        else:
            return ' '.join([process_type]) #, str(segment_type)])

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
        for tableau in inputs:
            random.shuffle(tableau)
            self.train_tableau(tableau)
        return self.errors

    def train_tableau(self, tableau):
        (grammatical_winner, computed_winner, correct) = self.evaluate(tableau)
        if correct:
            if len(computed_winner) == 1:
                pass
                #self.errors.append(0)
            else:
                computed_winner.remove(grammatical_winner)
                self.con.induce([grammatical_winner, computed_winner[0]])
                self.errors.append(''.join([str(grammatical_winner), '~t~', str(computed_winner)]))
                #self.train_tableau(tableau) # if you use this, think about error counting
        else:
            if numpy.random.random() <= self.induction_freq:
                self.con.induce([grammatical_winner, computed_winner[0]])
                #self.train_tableau(tableau)
            else:
                self.update(grammatical_winner, computed_winner[0])
            self.errors.append(''.join([str(grammatical_winner), '~', str(computed_winner)]))

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
    def __init__(self, feature_chart, input_file, num_negatives = 10, max_changes = 10,
                 processes = '[self.change_feature_value]', #'[self.delete, self.metathesize, self.change_feature_value, self.epenthesize]',
                 epenthetics = ['e', '?'], learning_rate = 0.1, num_trainings = 10,
                 aligned = True, tier_freq = .2, induction_freq = .1, constraint_parts = ['voi', 'son', 'word', 'round', 'back']):
        # input parameters
        self.feature_dict = FeatureDict(feature_chart)
        self.input_file = input_file
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = processes
        self.epenthetics = epenthetics

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
        self.training_runs = []
        self.testing_runs = []
        self.num_constraints_runs = []
        self.constraint_parts = constraint_parts
        with open(self.report, 'a') as f:
            f.write('\n'.join(['Feature Chart: ' + feature_chart, 'Input File: ' + input_file, '# Ungrammatical Candidates Generated: ' + str(num_negatives),
                               'Max Changes to Candidates: ' + str(max_changes), 'GEN processes: ' + processes, 'Epenthetic Segments: ' + str(epenthetics),
                               'Learning Rate: ' + str(learning_rate), 'Aligned Markedness Constraints: ' + str(aligned), 'Frequency of Tier Constraints: ' +
                               str(tier_freq), 'Frequency of Induction Upon Error: ' + str(induction_freq)]))

    def make_input(self):
        """Use Input class to convert input file to data structure or access previously saved data structure."""
        inputs = Input(self.feature_dict, self.input_file, self.num_negatives,
                       self.max_changes, self.processes, self.epenthetics)
        self.all_input = inputs.allinputs

    def divide_input(self):
        """Choose training set and test set."""
        test_size = int(len(self.all_input)/10)
        numpy.random.shuffle(self.all_input)
        self.test_input = self.all_input[:test_size]
        self.train_input = self.all_input[test_size:]

    def refresh_input(self, inputs):
        for tableau in inputs:
            for mapping in tableau:
                mapping.violations = numpy.array([1])
                mapping.harmony = None

    def refresh_con(self):
        self.alg.con.constraints = []
        self.alg.con.weights = numpy.array([0])

    def run(self):
        """Initialize HGGLA and do all training and testing iterations for this run."""
        self.alg = HGGLA(self.learning_rate, self.feature_dict, self.aligned, self.tier_freq, self.induction_freq)
        self.refresh_input(self.train_input)
        self.refresh_input(self.test_input)
        self.refresh_con()
        self.training_errors = []
        self.testing_errors = []
        self.num_constraints = []
        for i in range(self.num_trainings):
            self.train_HGGLA(i)
            self.test_HGGLA(i)
        self.training_runs.append(self.training_errors)
        self.testing_runs.append(self.testing_errors)
        self.num_constraints_runs.append(self.num_constraints)
        self.check_constraints()
        self.plot_constraints()

    def train_HGGLA(self, i):
        """Do one iteration through the training data."""
        errors = self.alg.train(self.train_input)
        self.training_errors.append(len(errors)/len(self.train_input))
        constraints_added = self.alg.con.constraints[self.num_constraints[-1]:] if i else self.alg.con.constraints
        self.num_constraints.append(len(self.alg.con.constraints))
        with open(self.report, 'a') as f:
            f.write(''.join(['\nErrors in training #', str(i), ': ', str(len(errors)), '\n'.join([error for error in errors])]))
            f.write(''.join(['\nConstraints added in training #', str(i), ': ',
                             str(len(constraints_added)), '\n'.join([str(c) for c in self.alg.con.constraints])]))

    def test_HGGLA(self, i):
        """Do one iteration through the testing data."""
        errors = []
        for tableau in self.test_input:
            winner = self.alg.test(tableau)
            if winner.grammatical == False:
                errors.append(str(winner))
        self.testing_errors.append(len(errors)/len(self.test_input))
        with open(self.report, 'a') as f:
            f.write(''.join(['\nErrors in testing #', str(i), ': ', str(len(errors)), '\n'.join([error for error in errors])]))

    def test_parameter(self, parameter, values):
        """If parameter is an input parameter, redo input for each value of the
        parameter. If parameter is an algorithm parameter, make and divide the
        input once, and then run the algorithm with each value of the
        parameter."""
        if parameter in ['feature_dict', 'input_file', 'num_negatives', 'max_changes', 'processes', 'epenthetics']:
            param = eval(parameter)
            for value in values:
                self.param = value
                self.make_input()
                self.divide_input()
                self.run()
        elif parameter in ['learning_rate', 'num_trainings', 'aligned', 'tier_freq']:
            param = eval(parameter)
            self.make_input()
            self.divide_input()
            for value in values:
                self.param = value
                self.run()
        else:
            raise AssertionError, 'Update parameter lists.'
        print ('tested ', parameter, 'on ', values, '\n')
        print('error percentage on last test of each run', [run[-1] for run in self.testing_runs],
              '\nnumber of constraints', [run[-1] for run in self.num_constraints_runs])
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
                f.write('--------Run ' + str(i) + '--------')
            self.run()
        print('ran program ', num_runs, ' times')
        print('error percentage on last test of each run', [run[-1] for run in self.testing_runs],
              '\nnumber of constraints', [run[-1] for run in self.num_constraints_runs])
        self.plot_errors()
        self.figs.close()

    def plot_constraints(self):
        """Plot the 20 highest weighted constraints and write all of them to a
        file with their weights, in descending order of weight."""
        constraints = [str(c) for c in self.alg.con.constraints]
        num_con = len(constraints)
        constraint_list = zip(self.alg.con.weights, constraints)
        constraint_list.sort()
        with open(self.report, 'a') as f:
            f.write('\n'.join([str(w) + '\t' + str(c) for (w, c) in constraint_list]))
        constraint_list = zip(*constraint_list)
        #ind = numpy.arange(num_con)
        ind = numpy.arange(20) if num_con > 20 else numpy.arange(num_con)
        height = .35
        #colors = ['r' if isinstance(c, Faithfulness) else 'y' for c in self.alg.con.constraints]
        pyplot.barh(ind, constraint_list[0][0:20]) #, color = colors
        pyplot.xlabel('Weights')
        pyplot.title('Constraint Weights')
        pyplot.yticks(ind+height/2., constraint_list[1][0:20])
        pyplot.subplots_adjust(left = .5) # make room for constraint names
        #pyplot.xticks(np.arange(0,81,10))
        #pyplot.legend( (p1[0], p2[0]), ('Markedness', 'Faithfulness') )
        #pyplot.clf()
        #pyplot.show()
        self.figs.savefig()
        pyplot.clf()

    def plot_errors(self, parameter = None, values = None):
        """Make two plots, one for training and one for testing. Each has
        iterations on the x axis and error percentage on the y axis. Plot a line
        for each run."""
        pyplot.subplots_adjust(left = .15)
        for item in [self.training_runs, self.testing_runs]:
            plots = []
            for run in item:
                plots.append(pyplot.plot(run))
            pyplot.xlabel('Iteration')
            pyplot.ylabel('Percent of Inputs Mapped to Incorrect Outputs')
            kind = 'Training' if item == self.training_runs else 'Testing'
            pyplot.title(kind + ' Error Rates')
            labels = [parameter + ' = ' + str(value) for value in values] if parameter else range(len(item))
            pyplot.legend(plots, labels, loc = 0)
            #pyplot.show()
            self.figs.savefig()
            pyplot.clf()

    def check_constraints(self):
        constraints_found = []
        with open(self.report, 'r') as f:
            text = f.readlines()
            for line in text:
                if line.startswith(('-','+')):
                    for part in self.constraint_parts: #there must be a better way than a double loop
                        if part in line:
                            constraints_found.append(line)
                            break
        #print constraints_found
        with open(self.report, 'a') as f:
            f.write('\n'.join(constraints_found))

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
    #learn1 = Learn('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    learner = Learn('feature_chart4.csv', 'input6.csv', processes = '[self.change_feature_value]', max_changes = 5, num_negatives = 50)
    #xval1 = CrossValidate('feature_chart3.csv', ['input3.csv'], tier_freq = 10)
    #xval2 = CrossValidate('feature_chart3.csv', ['input4.csv'], tier_freq = 10)
    #learner = Learn('TurkishFeaturesWithNA.csv', 'TurkishInput3.csv',
                         #num_trainings = 3, max_changes = 5, num_negatives = 15, tier_freq = 5, processes = '[self.change_feature_value]')
    #TurkishInput2 has the ~ inputs taken out, the variable inputs taken out, and deletion taken out.
    #TurkishInput1 is the same but deletion is still in.
    #same pattern for test files
    #TurkishInput3 is TurkishInput2 plus TurkishTest2
    #TurkishInput4 is TurkishInput3 with all underlying suffix vowels changed to i, and appropriate changes added.
    learner.test_performance()
