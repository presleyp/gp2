import cPickle, csv, copy, numpy, random
from mapping import Mapping, Change, ChangeNoStem

class Input:
    """Give input in the form of a csv file where each line is a mapping, with 0
    or 1 for ungrammatical or grammatical, then underlying form in segments,
    then surface form in segments, then semicolon-delimited list of changes from
    underlying form to surface form.  Ungrammatical mappings are optional; if
    you include them, the second line must be ungrammatical."""
    def __init__(self, feature_dict, input_file, remake_input, num_negatives, max_changes, processes,
        epenthetics, stem, gen_type):
        """Convert lines of input to mapping objects.  Generate
        ungrammatical input-output pairs if they were not already present in the
        input file."""
        print 'calling Input'
        self.feature_dict = feature_dict
        Operation = Change if stem else ChangeNoStem
        self.Generator = Gen if gen_type == 'random' else DeterministicGen
        self.gen_args = [num_negatives, max_changes, processes, epenthetics, Operation]
        try:
            assert remake_input == False
            saved_file = open('save-' + input_file, 'rb')
            self.allinputs = cPickle.load(saved_file)
            print 'read from file'
        except (IOError, AssertionError):
            self.allinputs = self.make_input(input_file)
            saved_file = open('save-' + input_file, 'wb')
            cPickle.dump(self.allinputs, saved_file)
        saved_file.close()
        print 'done making input'

    def find_stem(self, affixes):
        for i, affix in enumerate(affixes):
            for j, position in enumerate(affix):
                segment_class = []
                for segment in position:
                    segment_class.add(self.feature_dict.get_features_seg(segment))
                if len(segment_class) > 1:
                    affixes[i][j] = segment_class[0].intersection(*segment_class[1:])

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
                    gen = self.Generator(self.feature_dict, *self.gen_args)
                    negatives = gen.ungrammaticalize(mapping)
                    mapping.add_boundaries()
                    mapping.set_ngrams()
                    tableau = [mapping] + negatives
                    allinputs.append(tableau)
            return allinputs

class Gen:
    """Generates input-output mappings."""
    def __init__(self, feature_dict, num_negatives, max_changes, processes, epenthetics, Operation):
        self.feature_dict = feature_dict
        #self.major_features = self.feature_dict.major_features
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = eval(processes)
        self.epenthetics = epenthetics
        self.non_boundaries = copy.deepcopy(self.feature_dict.fd)
        del self.non_boundaries['|']
        self.Operation = Operation

    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly
        ungrammatical mappings with the same input."""
        negatives = []
        if mapping.ur.all() != mapping.sr.all():
            negatives.append(self.make_faithful_cand(mapping))
        while len(negatives) < self.num_negatives:
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
            new_mapping.stem = copy.copy(mapping.stem)
            for j in range(numpy.random.randint(0, self.max_changes + 1)):
                process = random.choice(self.processes)
                process(new_mapping)
            # Don't add a mapping if it's the same as the grammatical one.
            try:
                if new_mapping == mapping:
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

    def make_faithful_cand(self, mapping):
        new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
        new_mapping.stem = copy.copy(mapping.stem)
        new_mapping.add_boundaries()
        new_mapping.set_ngrams()
        return new_mapping

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
        # decide what to change
        locus = numpy.random.randint(0, len(mapping.sr))
        segment = mapping.sr[locus]
        # decide how many features to change
        num_to_change = numpy.random.zipf(2)
        if num_to_change > len(segment):
            num_to_change = numpy.random.randint(1, len(segment) + 1)
        # pick a segment to change to with that many different features
        closest_num = None
        new_segment = None
        changed_features = None
        for phone in self.non_boundaries.values():
            difference = segment - phone
            num_different = len(difference)
            if num_different == num_to_change:
                new_segment = phone
                changed_features = difference
                break
            else:
                if num_different != 0 and (closest_num == None or numpy.absolute(num_different - num_to_change) < numpy.absolute(closest_num - num_to_change)):
                    closest_num = num_different
                    new_segment = phone
                    changed_features = difference
        # change the segment
        mapping.sr[locus] = new_segment
        assert changed_features, 'no change made'
        for feature in changed_features:
            change = self.Operation(self.feature_dict, feature = feature, mapping = mapping, locus = locus)
            change.make_set()
            mapping.changes.append(change)

class DeterministicGen(Gen):
    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly
        ungrammatical mappings with the same input."""
        negatives = []
        # make faithful candidate
        if mapping.ur.all() != mapping.sr.all():
            negatives.append(self.make_faithful_cand(mapping))
        # make candidate with changed voicing
        voice = self.feature_dict.get_feature_number('voi')
        voicing = None
        for item in mapping.sr[-1]:
            if numpy.absolute(item) == voice:
                voicing = item
        if voicing:
            new_sr = copy.deepcopy(mapping.ur)
            locus = len(mapping.sr) - 1
            new_sr[locus].remove(voicing)
            new_sr[locus].add(-voicing)
            change = self.Operation(self.feature_dict, change_type = 'change', feature = voicing, mapping = mapping, locus = locus)
            change.make_set()
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), new_sr, [change]])
            new_mapping.add_boundaries()
            new_mapping.set_ngrams()
            negatives.append(new_mapping)
        # make vowel harmony candidate
        vocalic = self.feature_dict.get_feature_number('voc')
        back = self.feature_dict.get_feature_number('back')
        roundness = self.feature_dict.get_feature_number('round')
        last_vowel = None
        for i in range(len(mapping.ur)).reverse():
            if vocalic in mapping.ur[i]:
                last_vowel = i
                back = mapping.ur[i] & set([back, -back])
                roundness = mapping.ur[i] & set([roundness, -roundness])
                break
        for i in [[back], [roundness], [back, roundness]]:
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
            new_mapping.sr[i] -= set(i)
            j = [-x for x in i]
            new_mapping.sr[i] |= set(j)
            for feature in i:
                change = self.Operation(self.feature_dict, mapping = mapping, locus = last_vowel, feature = feature)
                change.make_set()
                new_mapping.changes.append(change)
            new_mapping.add_boundaries()
            new_mapping.set_ngrams()
            negatives.append(new_mapping)
        return negatives


