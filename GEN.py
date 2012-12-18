import cPickle, csv, copy, numpy, random, glob, os, errno
from mapping import Mapping, Change

class Input:
    """Give input in the form of a csv file where each line is a mapping, with 0
    or 1 for ungrammatical or grammatical, then underlying form in segments,
    then surface form in segments, then semicolon-delimited list of changes from
    underlying form to surface form.  Ungrammatical mappings are optional; if
    you include them, the second line must be ungrammatical."""
    def __init__(self, feature_dict, input_file, remake_input, gen_type, gen_args = None):
        """Convert lines of input to mapping objects.  Generate
        ungrammatical input-output pairs if they were not already present in the
        input file."""
        print 'calling Input'
        self.feature_dict = feature_dict
        self.Generator = RandomGen if gen_type == 'random' else DeterministicGen
        self.gen_args = gen_args
        input_dir = 'dir-' + input_file
        self.make_dir(input_dir)
        self.input_files = [x for x in glob.glob(input_dir + '/*')]
        if remake_input == False and self.input_files != []:
            #saved_file = open('save-' + input_file, 'rb')
            #self.allinputs = cPickle.load(saved_file)
            print 'read from file'
        else:
            self.input_files = self.make_input(input_file, input_dir)
            #saved_file = open('save-' + input_file, 'wb')
            #cPickle.dump(self.allinputs, saved_file)
        #finally:
            #saved_file.close()
        print 'done making input'

    def make_dir(self, input_dir):
        '''Make the directory if it doesn't already exist so that the files will have a place to go.'''
        try:
            os.makedirs(input_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise exception

    def find_stem(self, affixes):
        for i, affix in enumerate(affixes):
            for j, position in enumerate(affix):
                segment_class = []
                for segment in position:
                    segment_class.add(self.feature_dict.get_features_seg(segment))
                if len(segment_class) > 1:
                    affixes[i][j] = segment_class[0].intersection(*segment_class[1:])

    def make_input(self, input_file, input_dir):
        """Based on file of lines of the form "1,underlyingform,surfaceform,changes"
        create Mapping objects with those attributes.
        Bundle mappings into tableaux."""
        input_files = []
        with open(input_file, 'r') as f:
            fread = list(csv.reader(f))
            if self.gen_args:
                gen = self.Generator(self.feature_dict, *self.gen_args)
            else:
                gen = self.Generator(self.feature_dict)
            for i, line in enumerate(fread):
                mapping = Mapping(self.feature_dict, line)
                mapping.to_data()
                negatives = gen.ungrammaticalize(mapping)
                mapping.add_boundaries()
                mapping.set_ngrams()
                tableau = [mapping] + negatives
                tableau_file = ''.join([input_dir, '/', str(i), '.txt'])
                with open(tableau_file, 'w') as f:
                    cPickle.dump(tableau, f)
                input_files.append(tableau_file)
        return input_files

class Gen:
    """Generates input-output mappings."""
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict
        self.non_boundaries = copy.deepcopy(self.feature_dict.fd)
        del self.non_boundaries['|']

    def make_faithful_cand(self, mapping):
        if (mapping.ur != mapping.sr).any():
            new_mapping = Mapping(self.feature_dict, [False,
                                                      copy.deepcopy(mapping.ur),
                                                      copy.deepcopy(mapping.ur),
                                                      []])
            new_mapping.stem = copy.copy(mapping.stem)
            new_mapping.add_boundaries()
            new_mapping.set_ngrams()
            return [new_mapping]
        else:
            return []

class RandomGen(Gen):
    def __init__(self, feature_dict, num_negatives, max_changes, processes, epenthetics):
        super().__init__(self, feature_dict)
        self.num_negatives = num_negatives
        self.max_changes = max_changes
        self.processes = eval(processes)
        self.epenthetics = epenthetics

    def ungrammaticalize(self, mapping):
        """Given a grammatical input-output mapping, create randomly
        ungrammatical mappings with the same input."""
        negatives = self.make_faithful_cand(mapping)
        while len(negatives) < self.num_negatives:
            new_mapping = Mapping(self.feature_dict, [False, copy.deepcopy(mapping.ur), copy.deepcopy(mapping.ur), []])
            new_mapping.stem = copy.copy(mapping.stem)
            for _ in range(numpy.random.randint(0, self.max_changes + 1)):
                process = random.choice(self.processes)
                process(new_mapping)
            # Don't add a mapping if it's the same as the grammatical one.
            if new_mapping == mapping:
                continue
            # Don't add a mapping if it's the same as a previously
            # generated one.
            new_mapping.add_boundaries()
            if new_mapping not in negatives:
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
                if num_different != 0 and (closest_num == None or
                                           numpy.absolute(num_different - num_to_change) <
                                           numpy.absolute(closest_num - num_to_change)):
                    closest_num = num_different
                    new_segment = phone
                    changed_features = difference
        # change the segment
        mapping.sr[locus] = new_segment
        assert changed_features, 'no change made'
        for feature in changed_features:
            change = Change(self.feature_dict, feature = feature,
                                    mapping = mapping, locus = locus)
            change.make_set()
            mapping.changes.append(change)

class DeterministicGen(Gen):
    """Generates candidates that test the processes found in my Turkish corpus."""
    def ungrammaticalize(self, mapping):
        false_faithful_cand = self.make_faithful_cand(mapping)
        faithful_cand = false_faithful_cand[0] if false_faithful_cand else mapping
        changed_voice = self.change_voicing(faithful_cand)
        changed_vowels = self.change_vowels(faithful_cand)
        changed_both = [self.change_vowels(m) for m in changed_voice]
        changed_both_flat = [m for alist in changed_both for m in alist]
        changed_mappings = changed_voice + changed_vowels + changed_both_flat
        false_mappings = changed_mappings + false_faithful_cand
        assert false_faithful_cand == [] or false_mappings.count(false_faithful_cand[0]) <= 1
        mapping.add_boundaries()
        false_mappings = [m for m in false_mappings if m != mapping] #not working
        return false_mappings

    def change_voicing(self, base): # check for no copy bugs
        original = base.sr
        voicing = self.feature_dict.get_feature_number('voi')
        voiced_indices = [i for (i, segment) in enumerate(original) if voicing in segment]
        devoiced = self.make_new_mappings(base, voiced_indices, set([voicing]))
        voiceless_indices = [i for (i, segment) in enumerate(original) if -voicing in segment]
        voice_added = self.make_new_mappings(base, voiceless_indices, set([-voicing]))
        new_mappings = devoiced + voice_added
        new_mappings = [m for m in new_mappings if m != base]
        return new_mappings

    def change_vowels(self, base):
        vocalic = self.feature_dict.get_feature_number('voc')
        original = base.sr
        vowel_indices = [i for (i, segment) in enumerate(original) if vocalic in segment]
        new_mappings = []
        for i in vowel_indices:
            back = self.find_feature_value(base.ur[i], 'back')
            roundness = self.find_feature_value(base.ur[i], 'round')
            backround = back | roundness
            new_mappings += [self.make_new_mapping(base, i, item)
                            for item in [back, roundness, backround]]
        new_mappings = [m for m in new_mappings if m and m != base]
        return new_mappings

    def find_feature_value(self, segment, feature): # returns a set
        feature_number = self.feature_dict.get_feature_number(feature)
        return segment & set([feature_number, -feature_number])


    def make_new_mappings(self, old_mapping, loci, features):
        new_mappings = [self.make_new_mapping(old_mapping, locus, features)
                        for locus in loci]
        real_mappings = [m for m in new_mappings if m]
        return real_mappings

    def make_new_mapping(self, old_mapping, locus, features):
        new_sr = copy.deepcopy(old_mapping.sr)
        new_sr[locus] -= features
        for feature in features:
            new_sr[locus].add(-feature)
        try:
            self.feature_dict.get_segment(new_sr[locus])
        except IndexError:
            return []
        new_mapping = Mapping(self.feature_dict, [False,
                    copy.deepcopy(old_mapping.ur),
                    new_sr,
                    copy.deepcopy(old_mapping.changes)])
        new_mapping.stem = copy.copy(old_mapping.stem)
        for feature in features:
            change = Change(self.feature_dict, change_type = 'change', mapping =
                new_mapping, locus = locus, feature = feature)
            change.make_set()
            new_mapping.changes.append(change)
        new_mapping.add_boundaries()
        new_mapping.set_ngrams()
        return new_mapping



def open_tableau(file_list):
    """Generator that yields one tableau at a time."""
    for tableau_file in file_list:
        with open(tableau_file, 'r') as f:
            tableau = cPickle.load(f)
            yield tableau

