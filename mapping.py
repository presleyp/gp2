import numpy

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
                change = Change(self.feature_dict)
                changeparts = self.changes[i].split(' ')
                change.stem = changeparts.pop(0) if changeparts[0] == 'stem' else ''
                change.change_type = changeparts.pop(0)
                change.segment = changeparts.pop()
                try:
                    change.feature = changeparts.pop(0)
                    change.feature = self.feature_dict.get_feature_number(change.feature)
                    change.value = changeparts.pop(0)
                except IndexError:
                    pass
                change.make_set()
                self.changes[i] = change
                #segment = changeparts.pop()
                ##changeparts += self.feature_dict.major_features(self.feature_dict.get_features_seg(segment))
                #if changeparts[0] == 'metathesize':
                    #segment2 = changeparts.pop(-4)
                    ##segment2 = self.feature_dict.major_features(self.feature_dict.get_features_seg(segment2))
                    ##segment2 = [''.join([2, f]) for f in segment2]
                    ##changeparts += segment2
                #if changeparts[0] == 'change':
                    #value = changeparts.pop()
                    #polarity = '+' if value == 1 else '-'
                    #feature_name = changeparts.pop()
                    #feature = self.feature_dict.get_feature_number(feature_name)
                    #changeparts += [polarity, feature]
                #self.changes[i] = set(changeparts)

    def in_stem(self, locus):
        return True if self.stem[0] < locus < self.stem[1] else False

    def split(self, feature):
        if feature < 0:
            return ['-', numpy.absolute(feature)]
        else:
            return ['+', feature]

    def add_boundaries(self):
        boundary = self.feature_dict.get_features_seg('|')
        self.ur = numpy.hstack((boundary, self.ur, boundary))
        self.sr = numpy.hstack((boundary, self.sr, boundary))

    def __eq__(self, other):
        if len(self.sr) != len(other.sr):
            print 'self', self, 'other', other
        return numpy.equal(self.sr, other.sr).all()

    def __str__(self):
        ur = ''.join(self.feature_dict.get_segments(self.ur))
        sr = ''.join(self.feature_dict.get_segments(self.sr))
        changes = [str(change) for change in self.changes]
        return ', '.join([str(self.grammatical), ur, sr, str(changes)])

    def set_ngrams(self):
        self.ngrams = [self.get_ngrams(self.sr, 1), self.get_ngrams(self.sr, 2), self.get_ngrams(self.sr, 3)]

    def get_ngrams(self, word, n):
        starting_positions = range(len(word) + 1 - n)
        ngrams_in_word = [word[i:i + n] for i in starting_positions]
        return ngrams_in_word

class Change:
    def __init__(self, feature_dict): #TODO change to *kwargs and make make_set part of init?
        self.feature_dict = feature_dict
        self.change_type = None
        self.stem = ''
        self.feature = ''
        self.value = ''
        self.segment = ''
        self.context = 'change'
        self.change_to_faith = {'change': 'Ident', 'epenthesize': 'Dep', 'delete': 'Max', 'metathesize': 'Lin'}

    def make_set(self):
        self.set = set([self.change_type, self.stem, self.feature, self.value])
        self.set.discard('')

    def __str__(self):
        change_type = self.change_type if self.context == 'change' else self.change_to_faith[self.change_type]
        feature = self.feature_dict.get_feature_name(self.feature)
        return ''.join([change_type, ' ', self.stem, ' ', self.value, feature]) # TODO fix so I don't get double spaces

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

    def remove(self, to_delete):
        self.set.remove(to_delete)

    def add(self, other):
        self.set.add(other)


