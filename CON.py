import numpy, random, copy
#TODO privilege constraints where grams have same features, maybe also where values are the same

class Con:
    def __init__(self, feature_dict, tier_freq, aligned, stem):
        self.constraints = []
        self.weights = numpy.array([0]) # intercept weight
        self.feature_dict = feature_dict
        self.tier_freq = tier_freq
        self.aligned = aligned
        self.stem = stem
        self.i = 0

    def induce(self, winners):
        """Makes one new markedness and one new faithfulness constraint
        and initializes their weights, unless appropriate constraints
        cannot be found within 15 tries."""
        #print self.i
        assert len(self.weights) == len(self.constraints) + 1
        if random.random() < .5:
            self.make_constraint(Faithfulness, winners, self.feature_dict, self.stem)
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
        #self.num_features = self.feature_dict.num_features
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
                if len(tier_winners) >= desired_number and (tier_winners[0] - tier_winners[1]).any() and len(tier_winners[0]) == len(tier_winners[1]):
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

    #def dont_care(self, ngram):
        #self.dontcares = [random.sample(range(self.num_features), numpy.random.randint(0, self.num_features)) for segment in ngram]
        #self.cares = []
        #for segment in self.dontcares:
            #for feature in segment:
                #ngram[segment][feature] = 0

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
        #assert difference.any(), 'duplicates'
        protected_segment = random.choice(numpy.where(difference)[0])
        protected_feature = random.sample(difference[protected_segment], 1)[0]
        # pick ngram
        positions = range(self.gram)
        random.shuffle(positions)
        position = None
        for position_in_ngram in positions:
            pattern = copy.copy(base[protected_segment -
                                     position_in_ngram:protected_segment +
                                     self.gram - position_in_ngram])
            if len(pattern) == self.gram:
                self.constraint = pattern
                position = position_in_ngram
                break
        if self.constraint == None:
            print('gram', self.gram, 'base', base, 'protected seg',
                  protected_segment, 'winners', winners[0], winners[1], 'tier',
                  self.tier)
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
        """Flip a coin. If heads, the constraint will be made from the
        grammatical winner and will assign rewards. If tails, it will be made
        from the computed winner and will assign violations. Return the winner
        the constraint will be made from,
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
        if isinstance(other, Faithfulness):
            return False
        elif other == None:
            return self.constraint == None
        elif len(self.constraint) != len(other.constraint):
            return False
        else:
            return (numpy.equal(self.constraint, other.constraint)).all()

    def __str__(self):
        polarity = '+' if self.violation else '-'
        segments = []
        for segment in self.constraint:
            #natural_class = [k for k, v in self.feature_dict.fd.iteritems() if segment <= v]
            natural_class = [self.polarity(feature) +
                             self.feature_dict.get_feature_name(feature) for
                             feature in segment]
            natural_class = ''.join(['{', ','.join(natural_class), '}'])
            segments.append(natural_class)
        if self.tier:
            return ''.join([self.feature_dict.get_feature_name(self.tier), ' tier ', polarity, str(segments)])
        else:
            return ''.join([self.polarity(self.violation)] + [segment for segment in segments])

class Faithfulness:
    def __init__(self, winners, feature_dict, stem):
        """Find a change that exists in only one winner. Abstract away from some
        of its feature values, but not so much that it becomes equivalent to a
        change in the other winner. Make this a faithfulness constraint."""
        self.feature_dict = feature_dict
        self.stem = stem
        self.constraint = None
        self.base = copy.deepcopy(winners[1].changes)
        self.other = winners[0].changes
        assert type(self.base) == list
        random.shuffle(self.base)
        for change in self.base:
            if self.base.count(change) > self.other.count(change): #FIXME will this work?
                #print self.base.count(change), self.other.count(change)
                #print winners[1].changes, winners[0].changes
                self.constraint = change
                if not self.stem:
                    self.constraint.discard('stem')
                    self.remove_specific(change, change.value, winners)
                else:
                    for item in [change.value, change.stem]:
                        self.remove_specific(change, item, winners)
                break

    def remove_specific(self, change, item, winners):
        """Remove a source of specificity unless it makes the constraint
        incapable of distinguishing between the computed and grammatical
        winners."""
        if numpy.random.random() > .5:
            self.constraint.discard(item)
            violations_base = self.get_violation(winners[1])
            violations_other = self.get_violation(winners[0])
            if violations_base >= violations_other:
                self.constraint.add(item)

    def get_violation(self, mapping):
        """Finds the number of times the change referred to by the constraint
        occurs in the input-output pair."""
        violation = 0
        for change in mapping.changes:
            if self.constraint <= change:
                violation -= 1
        return violation

    def __eq__(self, other):
        if other == None:
            return self.constraint == None
        elif isinstance(other, Markedness):
            return False
        else:
            return self.constraint == other.constraint

    def __contains__(self, element):
        return element in self.constraint

    def __str__(self):
        self.constraint.context = 'faith'
        return str(self.constraint)

