import csv, numpy, copy

class FeatureDict:
    def __init__(self, feature_chart):
        """The attributes are: fd, a dictionary with strings as keys (segments) and 1D arrays of 1 and -1 as values (feature vectors).
        feature_names, a list of the strings (segments). num_features, the number of features that are defined. tiers, a list of the indices
        in a feature vector that hold features over which tiers can be made.
        The first three features in the feature chart supplied to the learner should be vocalic, consonantal, and sonorant."""
        self.fd = {}
        self.feature_names = None
        feature_indices = None
        with open(feature_chart, 'r') as fc:
            fcd = csv.reader(fc)
            first_line = fcd.next()
            segment = first_line.pop(0)
            self.feature_names = ['placeholder'] + first_line
            feature_indices = numpy.arange(1, len(first_line) + 1)
            for line in fcd:
                segment = line.pop(0)
                line = [int(item) for item in line]
                self.fd[segment] = set(numpy.array(line) * feature_indices)
                self.fd[segment].discard(0)
        self.tiers = self.init_tiers()

    def get_feature_number(self, feature_name):
        return self.feature_names.index(feature_name)

    def get_feature_name(self, feature_number):
        return self.feature_names[numpy.absolute(feature_number)]

    def get_features_seg(self, segment):
        """Given a string segment, returns a feature vector as a 1D numpy array."""
        return copy.copy(self.fd[segment])

    def get_features_word(self, word):
        """Given a string word, returns a 2D numpy array, with rows for segments
        and columns for features."""
        features = [self.get_features_seg(segment) for segment in word]
        return numpy.array(features)

    def get_segment(self, features):
        """Given a feature set, returns a string segment."""
        return [k for k, v in self.fd.iteritems() if numpy.equal(v, features).all()][0]

    def get_segments(self, word):
        """Given a 1D numpy array of feature sets, returns a string word."""
        return ''.join([self.get_segment(features) for features in word])

    def major_features(self, featureset):
        """Select only the first three features. These should be vocalic,
        consonantal, and sonorant."""
        majors = []
        for feature in ['voc', 'cons', 'son']:
            number = self.get_feature_number(feature)
            majors.add(number)
            majors.add(-number)
        return featureset & set(majors)

    def init_tiers(self):
        """Build a list of feature indices for the features that can have tiers
        and that are in the feature dictionary. Raises an error if there are
        none in the feature dictionary."""
        tiers = set()
        tier_names = ['voc', 'cons', 'nas', 'strid']
        for name in tier_names:
            try:
                tiers.add(self.get_feature_number(name))
            except ValueError:
                continue
        assert len(tiers) > 0, "feature chart doesn't support tiers"
        return tiers
