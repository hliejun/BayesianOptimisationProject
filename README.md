# Bayesian Optimisation Project
Using Bayesian Optimisation to determine which horse race to observe to make the most informed decision on which horse to bet on.

## Setup

### Installation
 > Dependency: Python 3.5.2, scikit-learn 

1. Install Anaconda3: https://www.continuum.io/downloads

2. Setup Jupyter: https://ipython.org/notebook.html

3. Install dependencies:

```
$ conda update conda
$ conda install scikit-learn
$ conda install numpy
    
```
4. Navigate to the project root directory and run:

```

$ jupyter notebook

```
5. Select our project .ipynb and view the project step-by-step.
    
### Approach
 > We will be modifying the Bayesian Optimization algorithm by fmfn:
https://github.com/fmfn/BayesianOptimization

1. We start off with a given set of horses, each with a set of features x.

2. We are assuming that we have a constraint of only visiting a single horse race every time we want to update our Gaussian Process model.

3. We will simulate this constraint by running Bayesian Optimisation on the race profiles (x values of horses in the race) and using the different acquisition functions, we determine which race results to release to update our model.

4. We will repeat this for a number of iterations and review the results of our experiment.
