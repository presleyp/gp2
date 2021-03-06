import numpy
    #TODO change input file to -voi instead of voi -1 format

class Mapping:
    def __init__(self, feature_dict, line):
        """Each input-output mapping is an object with attributes: grammatical (is it grammatical?),
        ur (underlying form), sr (surface form), changes (operations to get from ur to sr),
        violations (of constraints in order), harmony (violations times constraint weights), and self.ngrams, which is a list of lists,
        first of its unigrams, then of its bigrams, and then of its trigrams.
        ur and sr are numpy arrays, with segments as rows and features as columns."""
        self.grammatical = line[0]
        self.ur = line[1]
        self.sr = line[2]
        self.changes = line[3]
        self.violations = numpy.array([1]) # intercept
        self.harmony = None
        #self.meaning = line[4] if len(line) == 5 else None
        self.feature_dict = feature_dict
        self.ngrams = None
        self.stem = None

    def to_data(self):
        self.grammatical = bool(int(self.grammatical))
        morphemes = self.ur.split('+')
        beginning = len(morphemes[0])
        end = beginning + len(morphemes[1])
        self.stem = (beginning, end)
        self.ur = ''.join(morphemes)
        self.ur = self.feature_dict.get_features_word(self.ur)
        self.sr = self.feature_dict.get_features_word(self.sr)
        if self.changes == 'none':
            self.changes = []
        else:
            self.changes = self.changes.split(';')
            for i in range(len(self.changes)):
                changeparts = self.changes[i].split(' ')
                stem = changeparts.pop(0) if changeparts[0] == 'stem' else ''
                change = Change(self.feature_dict, change_type = changeparts[0])
                change.stem = stem
                segment = changeparts.pop()
                try:
                    feature = changeparts.pop(1)
                    change.feature = self.feature_dict.get_feature_number(feature)
                    value = changeparts.pop(1)
                    change.value = '+' if value == '1' else '-'
                except IndexError:
                    pass
                change.make_set()
                self.changes[i] = change

    def in_stem(self, locus): #TODO check that <= not needed
        return True if self.stem[0] <= locus < self.stem[1] else False

    def split(self, feature):
        if feature < 0:
            return ['-', numpy.absolute(feature)]
        else:
            return ['+', feature]

    def add_boundaries(self):
	'''Adds | feature set to both ends of ur and sr if it's not
        already there.'''
        boundary = self.feature_dict.get_features_seg('|')
	if boundary != self.ur[0]:
	    self.ur = numpy.hstack((boundary, self.ur, boundary))
            self.sr = numpy.hstack((boundary, self.sr, boundary))

    def __eq__(self, other):
        if len(self.sr) != len(other.sr):
            return False
        return numpy.equal(self.sr, other.sr).all()

    def __ne__(self, other):
	if self == other:
	    return False
	else:
	    return True

#FIXME i want to affect how if mapping in collection is done
    #def __contains__(self, collection):
        #for item in collection:
            #if self == item:
                #return True
        #return False

    def __str__(self):
        ur = ''.join(self.feature_dict.get_segments(self.ur))
        sr = ''.join(self.feature_dict.get_segments(self.sr))
        changes = []
        for change in self.changes:
            change.context = 'change'
            changes.append(str(change))
        return ', '.join([str(self.grammatical), ur, sr, str(changes)])

    def set_ngrams(self):
        self.ngrams = [self.get_ngrams(self.sr, 1), self.get_ngrams(self.sr, 2), self.get_ngrams(self.sr, 3)]

    def get_ngrams(self, word, n):
        starting_positions = range(len(word) + 1 - n)
        ngrams_in_word = [word[i:i + n] for i in starting_positions]
        return ngrams_in_word

class Change:
    def __init__(self, feature_dict, change_type = 'change', feature = '', mapping = '', locus = ''):
        self.feature_dict = feature_dict
        if feature:
            self.feature = numpy.absolute(feature)
            self.value = '+' if feature > 0 else '-'
            self.change_type = 'change'
        else:
            self.feature = ''
            self.value = ''
            self.change_type = change_type
        self.stem = ''
        if mapping:
            if mapping.in_stem(locus):
                self.stem = 'stem'
        self.context = 'change'
        self.change_to_faith = {'change': 'Ident', 'epenthesize': 'Dep', 'delete': 'Max', 'metathesize': 'Lin'}

    def make_set(self):
        self.set = set([self.change_type, self.stem, self.feature, self.value])
        self.set.discard('')

    def __str__(self):
        change_type = self.change_type if self.context == 'change' else self.change_to_faith[self.change_type]
        feature = self.feature_dict.get_feature_name(self.feature)
        stem = self.stem if self.stem in self.set else ''
        value = self.value if self.value in self.set else ''
        return ''.join([change_type, ' ', stem, ' ', value, feature])

    def __eq__(self, other):
        if other == None:
            return self.set == None
        else:
            return self.set == other.set

    def __le__(self, other):
        return self.set <= other.set

    def __ge__(self, other):
        return self.set >= other.set

    def __contains__(self, element):
        return element in self.set

    def discard(self, to_delete):
        self.set.discard(to_delete)

    def add(self, other):
        self.set.add(other)
