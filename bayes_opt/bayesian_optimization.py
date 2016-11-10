import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .helpers import UtilityFunction, unique_rows, PrintLog, acq_max


class BayesianOptimization(object):

    def __init__(self, parameter_bounds, gp_kernel=Matern(), verbose=1):
        """
        :param parameter_bounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the original dictionary
        self.parameter_bounds = parameter_bounds

        # Get the name of the parameters
        self.keys = list(parameter_bounds.keys())

        # Find number of parameters
        self.dim = len(parameter_bounds)

        # Create an array with parameters bounds
        bounds = []
        for key in self.parameter_bounds.keys():
            bounds.append(self.parameter_bounds[key])
        self.bounds = np.asarray(self.bounds)

        # Initialization flag
        self.initialized = False

        # Numpy array placeholders
        self.X = None
        self.Y = None

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=gp_kernel,
            n_restarts_optimizer=25,
        )

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)

        # Output dictionary
        self.res = {'max': {'max_val': None,
                            'max_params': None},
                    'all': {'values': [], 'params': []}}
        # Output dictionary

        # Verbose
        self.verbose = verbose

    def initialize(self, points):
        """
        Method to introduce labelled points

        :param points: np.array with columns
        (target, {list of columns matching self.keys})

        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281

        label must be in column 0

        :param y_column:
        column index of target

        :return:
        """

        self.X = np.delete(points, 0, 1)
        self.Y = points[0]
        self.initialized = True

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """

        # Update the internal object stored dict
        self.parameter_bounds.update(new_bounds)

        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.parameter_bounds.keys()):

            # Reset all entries, even if the same.
            self.bounds[row] = self.parameter_bounds[key]

    def maximize(self,
                 n_batches,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param n_batches:

        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.

        :param kappa:

        :param xi:

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        Returns
        -------
        :return: Nothing
        """
        n_iter = len(n_batches)

        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        y_max = self.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.
        selected_batch = acq_max(ac=self.util.utility,
                                 gp=self.gp,
                                 batches=n_batches[0],
                                 y_max=y_max,
                                 bounds=self.bounds)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)

        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):

            # TO-DO: select X and Y from selected batch
            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, selected_batch.x))
            self.Y = np.append(self.Y, selected_batch.y)

            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]

            # Maximize acquisition function to find next probing point
            selected_batch = acq_max(ac=self.util.utility,
                                     gp=self.gp,
                                     batches=n_batches[i],
                                     y_max=y_max,
                                     bounds=self.bounds)

            # Print stuff
            if self.verbose:
                self.plog.print_step(self.X[-1], self.Y[-1])

            # Keep track of total number of iterations
            self.i += 1

            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved

        :param file_name: name of file where points will be saved in csv format

        :return: None
        """

        points = np.hstack((self.X, np.expand_dims(self.Y, axis=1)))
        header = ', '.join(self.keys + ['target'])
        np.savetxt(file_name, points, header=header, delimiter=',')
