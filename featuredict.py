import csv, numpy, copy, random

class FeatureDict:
    def __init__(self, feature_chart):
        """The attributes are: fd, a dictionary with strings as keys (segments) and 1D arrays of 1 and -1 as values (feature vectors).
        feature_names, a list of the strings (segments). num_features, the number of features that are defined. tiers, a list of the indices
        in a feature vector that hold features over which tiers can be made.
        The first three features in the feature chart supplied to the learner should be vocalic, consonantal, and sonorant."""
        self.fd = {}
        with open(feature_chart, 'r') as fc:
            fcd = csv.reader(fc)
            for i, line in enumerate(fcd):
                segment = line.pop(0)
                if i == 0:
                    self.feature_names = line
                if i > 0:
                    self.num_features = len(line)
                    for j in range(len(line)):
                        line[j] = int(line[j])
                        #line[j] = -1 if line[j] == 0 else line[j]
                    self.fd[segment] = numpy.array(line)
        self.tiers = self.init_tiers()

    def get_features_seg(self, segment):
        """Given a string segment, returns a feature vector as a 1D numpy array."""
        return copy.copy(self.fd[segment])

    def get_features_word(self, word):
        """Given a string word, returns a 2D numpy array, with rows for segments
        and columns for features."""
        features = [self.get_features_seg(segment)
        for segment in word]
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
        majors = self.feature_names[0:3]
        features = copy.copy(featureset[0:3])
        major_features = []
        for i in range(3):
            major_features.append(':'.join([majors[i], str(features[i])]))
        return major_features

    def init_tiers(self):
        """Build a list of feature indices for the features that can have tiers
        and that are in the feature dictionary. Raises an error if there are
        none in the feature dictionary."""
        tiers = []
        tier_names = ['vocalic', 'consonantal', 'nasal', 'strident']
        for name in tier_names:
            try:
                tiers.append(self.feature_names.index(name))
            except ValueError:
                pass
        assert len(tiers) > 0, "feature chart doesn't support tiers"
        return tiers

