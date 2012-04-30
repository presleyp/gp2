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
        self.meaning = line[4] if len(line) == 5 else None
        self.feature_dict = feature_dict
        self.ngrams = None

    def to_data(self):
        self.grammatical = bool(int(self.grammatical))
        self.ur = self.feature_dict.get_features_word(self.ur)
        self.sr = self.feature_dict.get_features_word(self.sr)
        if self.changes == 'none':
            self.changes = []
        else:
            self.changes = self.changes.split(';')
            for i in range(len(self.changes)):
                changeparts = self.changes[i].split(' ')
                segment = changeparts.pop()
                #changeparts += self.feature_dict.major_features(self.feature_dict.get_features_seg(segment))
                if changeparts[0] == 'metathesize':
                    segment2 = changeparts.pop(-4)
                    #segment2 = self.feature_dict.major_features(self.feature_dict.get_features_seg(segment2))
                    #segment2 = [''.join([2, f]) for f in segment2]
                    #changeparts += segment2
                if changeparts[0] == 'change':
                    value = changeparts.pop()
                    polarity = '+' if value == 1 else '-'
                    feature = changeparts.pop()
                    feature = self.feature_dict.feature_names.index(feature)
                    changeparts += [polarity, feature]
                self.changes[i] = set(changeparts)

    def split(self, feature):
        if feature < 0:
            return ['-', numpy.absolute(feature)]
        else:
            return ['+', feature]

    def add_boundaries(self):
        boundary = self.feature_dict.fd['|']
        self.ur = numpy.hstack((boundary, self.ur, boundary))
        self.sr = numpy.hstack((boundary, self.sr, boundary))

    def __eq__(self, other):
        if len(self.sr) != len(other.sr):
            print 'self', self, 'other', other
        return numpy.equal(self.sr, other.sr).all()

    def __str__(self):
        ur = ''.join(self.feature_dict.get_segments(self.ur))
        sr = ''.join(self.feature_dict.get_segments(self.sr))
        return ', '.join([str(self.grammatical), ur, sr, str(self.changes)])

    def set_ngrams(self):
        self.ngrams = [self.get_ngrams(self.sr, 1), self.get_ngrams(self.sr, 2), self.get_ngrams(self.sr, 3)]

    def get_ngrams(self, word, n):
        starting_positions = range(len(word) + 1 - n)
        ngrams_in_word = [word[i:i + n] for i in starting_positions]
        return ngrams_in_word

