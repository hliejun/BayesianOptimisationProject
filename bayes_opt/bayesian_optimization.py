import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .helpers import UtilityFunction, get_unique_rows, PrintLog, acq_max


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
        self.bounds = []
        
        for key in self.parameter_bounds.keys():
            self.bounds.append(self.parameter_bounds[key])
        self.bounds = np.asarray(self.bounds)

        # Initialization flag
        self.initialized = False
        self.hasSetup = False

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

        self.selected_groups_scores = []
        self.average_batch_scores = []

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

        :return:
        """
        if not self.hasSetup:
            raise RuntimeError("BO has not been set up yet.")
        
        # print("ORIGINAL: ", points)
        
        self.X = np.delete(points, 0, 1)
        self.Y = points[0]
        
        # print("X: ", self.X)
        # print("Y: ", self.Y)

        # Update GP with unique rows of X to prevent GP from breaking
        unique_rows = get_unique_rows(self.X)
        self.gp.fit(self.X[unique_rows - 1], self.Y[unique_rows - 1])

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

    def setup(self, acq='poi', kappa=2.576, xi=0.0, **gp_params):
        """
        Customize BO

        :param acq:
            Acquisition function to be used,
            defaults to Probability of Improvement.

        :param kappa:
            For Upper Confidence Bound

        :param xi:
            For Expected Improvement and Probability of Improvement

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        :return:
        """
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        self.hasSetup = True

    def minimize(self, n_batches):
        """
        Main optimization method.

        :param n_batches:
            np.array [batch][group][sample][feature]

        :return: Nothing
        """
        n_iter = len(n_batches)

        # Reset timer
        self.plog.reset_timer()

        y_max = self.Y.max()

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
            # Find argmax of the acquisition function.
            results = acq_max(ac=self.util.utility,
                              gp=self.gp,
                              groups=n_batches[i],
                              y_max=y_max,
                              bounds=self.bounds)
            selected_group, selected_group_score, average_batch_score = results

            self.selected_groups_scores.append(selected_group_score)
            self.average_batch_scores.append(average_batch_score)

            # Append most recently generated values to X and Y arrays
            new_Xs = selected_group[:, 1:]
            new_Ys = selected_group[:, 0]
            self.X = np.vstack((self.X, new_Xs))
            self.Y = np.append(self.Y, new_Ys)

            # Update GP
            unique_rows = get_unique_rows(self.X) - 1
            self.gp.fit(self.X[unique_rows], self.Y[unique_rows])

            # Update maximum value to search for next probe point.
            highest_new_y = new_Ys.max()
            if highest_new_y > y_max:
                y_max = highest_new_y

            # Print stuff
            if self.verbose:
                for j in range(len(new_Ys)):
                    self.plog.print_step(new_Xs[j], new_Ys[j])

            # Keep track of total number of iterations
            self.i += 1

            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
                
            self.res['all']['values'].extend(new_Ys)
            for j in range(len(new_Xs)):
                self.res['all']['params'].append(
                    dict(zip(self.keys, new_Xs[j])))

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
