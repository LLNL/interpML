import numpy as np
from optparse import OptionParser
import os

class Sampler:

    __slots__ = ['alpha', 'd', 'omega', 'rng', 'sigma', 'test_range', 'spacing']

    def __init__(self, rng, d, frequency=1.0, skewness=0.0, std_dev=1.0,
                 test_range=1.0, spacing_kind='sob'):
        """ Initializer for Sampler class.

        Args:
            rng (generator): numpy random number generator; seed from main  
                determines a unique generator; enables reproducibility

            d (int): The dimension of the sample

            frequency (float): The frequency of the response surface
                (defaults to 1.0)

            skewness (float): A positive number defining how skewed
                the sample will be. 0 = no skew (default), large numbers =
                larger skew.

            std_dev (float): The clustering of sample points (defaults to 1.0)

            test_range (float): The range of the test points is
                [-test_range, test_range] (defaults to 1.0).

            spacing kind (string): the spacing stragegy used for sampling
                supported types: 
                    'sob' = Sobol sequence
                    'lhc' = Latin hyper-cube
                    'uni' = uniform sampling

        """

        # Check inputs
        if not isinstance(d, int) or d < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(frequency, float) or frequency < 0:
            raise ValueError("frequency must be a nonnegative float")
        if not isinstance(skewness, float) or skewness < 0:
            raise ValueError("skewness must be a nonnegative float")
        if not isinstance(std_dev, float) or std_dev <= 0:
            raise ValueError("standard deviation must be a positive float")
        if not isinstance(test_range, float) or test_range <= 0:
            raise ValueError("test range must be a positive float")
        if spacing_kind not in ['sob', 'lhc', 'uni']:
            raise ValueError("spacing must be sob (Sobol), lhc (Latin hyper-cube), or uni (uniform)")
        
        # Set values
        self.rng = rng  # pointer to numpy random number generator
        self.d = d
        self.alpha = skewness
        self.sigma = std_dev
        self.omega = frequency
        self.test_range = test_range
        self.spacing = spacing_kind
        return

    def sample_points(self, n):
        """ Draw n samples from a distribution with variance sigma.

        Args:

            n (int): number of samples to draw

        Returns:
            numpy.ndarray: A n X d matrix of n samples in [-1, 1]^d.

        """

        from scipy.stats import qmc

        # Initialize the output
        X = np.zeros((n, self.d))
        Y = np.zeros(n)
        if self.spacing == 'sob':
            # Use a scrambled Sobol sequence (returns points in [0,1]^d)
            sampler = qmc.Sobol(self.d, scramble=True, seed=self.rng)
            X[:, :] = sampler.random(n)
        elif self.spacing == 'lhc':
            # Use a scrambled Latin Hypercube strategy (additional keyword options available in scipy)
            sampler = qmc.LatinHypercube(self.d, scramble=True, seed=self.rng)
            X[:, :] = sampler.random(n)
        elif self.spacing == 'uni':
            # Draw each coordinate from [0,1] uniformly
            X = self.rng.uniform(size=(n,self.d)) 
        else:
            print("Houston, we have a problem.")
            exit()
        
        # Transform X from [0,1]^d to [-1,1]^d
        X = (2 * X - 1)
        # Generate skews
        X = np.dot(X, self.gen_skew())
        # Generate response values
        for i, xi in enumerate(X):
            Y[i] = self.rf(xi[:])
        return X, Y

    def test_points(self, n):
        """ Draw n test points.

        Args:

            n (int): number of test samples to draw

        """

        X = self.rng.random((n, self.d))
        X = (2 * X - 1) * self.test_range
        Y = np.zeros(n)
        for i, xi in enumerate(X):
            Y[i] = self.rf(xi)
        return X, Y

    def gen_skew(self):
        """ Generate the skew matrix.

        Returns:

            2d numpy.ndarray: The skew matrix Q, which is the identity
            times skew factors: Q[i, i] = eye[i] * alpha**i.

        """

        Q = np.eye(self.d)
        for i in range(self.d):
            Q[i, i] = np.exp(-i * self.alpha / (self.d + 1))
        return Q

    def rf(self, x):
        """ Evaluates the response function f.

        Args:

            x (np.ndarray): the input value to evaluate

        Returns:

            float: the corresponding response value

        """

        ## mixture of paraboloid + cosin product, shifted by 0.5 in each coord
        z = x - 0.5*np.ones_like(x)
        return 0.5 * (np.sum(z[:] ** 2) / x.size -
                      np.prod(np.cos(2 * np.pi * self.omega * z[:])))

        ## mixture of paraboloid + cosin product
        # return 0.5 * (np.sum(x[:] ** 2) / x.size -
        #               np.prod(np.cos(2 * np.pi * self.omega * x[:])))

        ## paraboloid
        # return np.sum(x[:] ** 2)
    

        ## Griewank
        # # first, scale by 100 since domain is [-1,1]^d
        # x *= 100

        # term_1 = (1. / 4000.) * sum(x[:] ** 2)
        # term_2 = 1.0
        # for i, xi in enumerate(x):
        #     term_2 *= np.cos(xi) / np.sqrt(i + 1)

        # return 1. + term_1 - term_2
       

        ## sinusoidal function (converges to 0 a.e. as d->infty)
        # return np.prod(np.sin(2 * np.pi * self.omega * x[:]))

# Not part of Sampler class
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


if __name__ == "__main__":
    """ Driver code """

    usage = "%prog [options]"
    parser = OptionParser(usage)
    parser.add_option( "--spacing", dest="spacing", type=str, default='sob', 
        help="Spacing used for sampling: sob (Sobol, default), lhc (Latin hyper-cube), or uni (uniform)")
    parser.add_option( "--d", dest="d", type=int, default=2, 
        help="Dimension of space from which samples are drawn.  Default 2.")
    parser.add_option( "--frequency", dest="frequency", type=float, default=1.0, 
        help="Frequency to be used in Sampler class constructor.  Default 1.0.")
    parser.add_option( "--skewness", dest="skewness", type=float, default=0.0, 
        help="Skewness to be used in Sampler class constructor.  Default 0.0.")
    parser.add_option( "--std_dev", dest="std_dev", type=float, default=1.0, 
        help="Standard deviation to be used in Sampler class constructor.  Default 1.0.")
    parser.add_option( "--n", dest="n", type=int, default=10, 
        help="Number of samples to draw.  Default 10.")
    parser.add_option("--seed", dest="seed", type=int, default=0,
        help="Value passed as global seed to random number generator.  Default 0.")
    
    (options, args) = parser.parse_args()


    print("Options selected:")
    print("  spacing   =", options.spacing)
    print("  dimension =", options.d)
    print("  frequency =", options.frequency)
    print("  skewness  =", options.skewness)
    print("  std_dev   =", options.std_dev)
    print("  num samp  =", options.n)
    print("  seed      =", options.seed)
    print()

    globalseed = options.seed
    rng = np.random.default_rng(globalseed) 

    my_sampler = Sampler(rng, options.d, frequency=options.frequency, 
                         skewness=options.skewness, std_dev=options.std_dev, spacing_kind=options.spacing)
    X, Y = my_sampler.sample_points(options.n)

    assert(X.shape == (options.n, options.d))
    assert(np.all(X > -1) and np.all(X < 1))
    assert(Y.shape == (options.n, ))
    ## use below assertion if Y values have known bounds
    # assert(np.all(Y > -1) and np.all(Y < 1)) 
    assert os.getcwd()[-7:] == 'scripts', "Run data_gen.py from the scripts sub-directory to ensure data is saved in correct location"
        
    output_dir = os.getcwd() + '/../data_store/'
    mkdir_p(output_dir)

    outfilename = output_dir  \
            + "sample_sp_" + str(options.spacing) \
            + "_d_" + str(options.d) \
            + "_fr_" + str(options.frequency) \
            + "_sk_" + str(options.skewness) \
            + "_sd_" + str(options.std_dev) \
            + "_n_" + str(options.n) \
            + "_seed_" + str(options.seed) \
            + ".npy"
    print("Collected sample, saving to", outfilename)
    
    outdata = np.concatenate((X, Y.reshape((options.n,1))), axis=1)

    np.save(outfilename, outdata)

    # # print(outdata)
    # import pandas as pd
    # odf = pd.DataFrame(outdata)
    # print(odf.describe())

