import numpy as np
"""
Descriptive statistics refactored as O(n).

Welford's algorithm computes the sample variance incrementally.

See also: http://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
"""
class OnlineVariance(object):

    """
    TODO describe constructor
    """
    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2, self.variance = ddof, 0, 0.0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    """
    add a new scalar value as observation to the dataset
    """
    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)
