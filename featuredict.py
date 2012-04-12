import csv, numpy, copy
class FeatureDict:
    def __init__(self, feature_chart):
        """The attributes are: fd, a dictionary with strings as keys (segments) and 1D arrays of 1 and -1 as values (feature vectors).
        feature_names, a list of the strings (segments). num_features, the number of features that are defined. tiers, a list of the indices
        in a feature vector that hold features over which tiers can be made.
        The first three features in the feature chart supplied to the learner should be vocalic, consonantal, and sonorant."""
        self.fd = {}
        self.feature_names = None
        self.num_features = None
        self.feature_indices = None
        with open(feature_chart, 'r') as fc:
            fcd = csv.reader(fc)
            for i, line in enumerate(fcd):
                segment = line.pop(0)
                if i == 0:
                    self.feature_names = ['placeholder'] + line #this way, indices of feature names are the numbers I assign to them, which don't include 0
                    self.num_features = len(line)
                    self.feature_indices = numpy.arange(1, self.num_features + 1)
                if i > 0:
                    line = [int(item) for item in line]
                    self.fd[segment] = set(numpy.array(line) * self.feature_indices)
                    self.fd[segment].discard(0)
        self.tiers = self.init_tiers()

    def get_features_seg(self, segment):
        """Given a string segment, returns a feature vector as a 1D numpy array."""
        return copy.copy(self.fd[segment])

    def get_features_word(self, word):
        """Given a string word, returns a 2D numpy array, with rows for segments
        and columns for features."""
        features = [self.get_features_seg(segment) for segment in word]
        return numpy.array(features)

    def get_segment(self, features):
        """Given a feature vector, returns a string segment."""
        return [k for k, v in self.fd.iteritems() if numpy.equal(v, features).all()][0]

    def get_segments(self, word):
        """Given a 2D numpy array of feature values, returns a string word."""
        return ''.join([self.get_segment(features) for features in word])

    def major_features(self, featureset):
        """Select only the first three features. These should be vocalic,
        consonantal, and sonorant."""
        features = copy.copy(featureset)
        majors = features & set([1,2,3,-1,-2,-3])
        major_features = set()
        for feature in major_features:
            major_features |= float(feature)
        return major_features

    def init_tiers(self):
        """Build a list of feature indices for the features that can have tiers
        and that are in the feature dictionary. Raises an error if there are
        none in the feature dictionary."""
        tiers = set()
        tier_names = ['vocalic', 'consonantal', 'nasal', 'strident']
        for name in tier_names:
            try:
                tiers.add(self.feature_names.index(name))
            except ValueError:
                pass
        assert len(tiers) > 0, "feature chart doesn't support tiers"
        return tiers

