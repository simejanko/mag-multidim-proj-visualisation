""" Statistic utilities. Full source found at https://github.com/biolab/orange3-bioinformatics. """
import math
import threading

def _lngamma(z):
    x = 0
    x += 0.1659470187408462e-06 / (z + 7)
    x += 0.9934937113930748e-05 / (z + 6)
    x -= 0.1385710331296526 / (z + 5)
    x += 12.50734324009056 / (z + 4)
    x -= 176.6150291498386 / (z + 3)
    x += 771.3234287757674 / (z + 2)
    x -= 1259.139216722289 / (z + 1)
    x += 676.5203681218835 / z
    x += 0.9999999999995183
    
    return math.log(x) - 5.58106146679532777 - z + (z - 0.5) * math.log(z + 6.5)
        

class LogBin(object):
    _max = 2
    _lookup = [0.0, 0.0]
    _max_factorial = 1
    _lock = threading.Lock()

    def __init__(self, max=1000):
        self._extend(max)

    @staticmethod
    def _extend(max):
        with LogBin._lock:
            if max <= LogBin._max:
                return
            for i in range(LogBin._max, max):
                if i > 1000:  # an arbitrary cuttof
                    LogBin._lookup.append(LogBin._logfactorial(i))
                else:
                    LogBin._max_factorial *= i
                    LogBin._lookup.append(math.log(LogBin._max_factorial))
            LogBin._max = max

    def _logbin(self, n, k):
        if n >= self._max:
            self._extend(n + 100)
        if k < n and k >= 0:
            return self._lookup[n] - self._lookup[n - k] - self._lookup[k]
        else:
            return 0.0

    @staticmethod
    def _logfactorial(n):
        if (n <= 1):
            return 0.0
        else:
            return _lngamma(n + 1)

class Hypergeometric(LogBin):
    """ `Hypergeometric distribution <http://en.wikipedia.org/wiki/Hypergeometric_distribution>`_ is
    a discrete probability distribution that describes the number of successes in a sequence of n draws
    from a finite population without replacement.
    """

    def __call__(self, k, N, m, n):
        """ If m out of N experiments are positive return the probability
        that k out of n experiments are positive using the hypergeometric
        distribution (i.e. return bin(m, k)*bin(N-m, n-k)/bin(N,n)
        where bin is the binomial coefficient).
        """
        if k < max(0, n + m - N) or k > min(n, m):
            return 0.0
        try:
            return min(math.exp(self._logbin(m, k) + self._logbin(N - m, n - k) - self._logbin(N, n)), 1.0)
        except (OverflowError, ValueError) as er:
            print(k, N, m, n)
            raise

    def p_value(self, k, N, m, n):
        """ 
        The probability that k or more tests are positive.
        """

        if min(n, m) - k + 1 <= k:
            # starting from k gives the shorter list of values
            return sum(self.__call__(i, N, m, n) for i in range(k, min(n,m)+1))
        else:
            value = 1.0 - sum(self.__call__(i, N, m, n) for i in (range(k)))
            # if the value is small it is probably inexact due to the limited
            # precision of floats, as for example  (1-(1-1e-20)) -> 0
            # if so, compute the result without substraction
            if value < 1e-3:  # arbitary threshold
                # print "INEXACT", value, sum(self.__call__(i, N, m, n) for i in range(k, min(n,m)+1))
                return sum(self.__call__(i, N, m, n) for i in range(k, min(n, m)+1))
            else:
                return value