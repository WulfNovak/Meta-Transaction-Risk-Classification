# Meta-Transaction-Risk-Classification
The following notebook exists to achieve the following:
- Familiarize self with Python and scikit learn
- Use Halving Grid Search and Bayesian Optimization to select hyperparameters
- Practice coding Voting and Stacking Ensemble models in Python

Some conclusions from this notebook:
- Halving grid search is computationally efficient and effective and may be preferable if training time should be minimized.
- Bayesian Optimization is less computationally efficient, but may be better for achieving a global optimum.
- Voting and Stacking Ensemble models are not effective when given input ensemble models, particularly when all input models are built on the same dataset.
