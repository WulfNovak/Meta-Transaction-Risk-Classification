# Meta-Transaction-Risk-Classification
The following notebook exists to achieve the following:
- Explore Meta Transaction Data and build machine learning models to classify categorical risk level.
- Using scikit-learn, create model pipeline to test different model types.
- Use Halving Grid Search and Bayesian Optimization to select hyperparameters
- Combine models into voting and stacking ensemble models, contrast resulting accuracy.

Noted Results:
- Deduced that data is synthetic
- Multicollinearity between several key variables
  -  Removed from analysis
- Halving Grid Search resulted in *marginally* better Light GBM models
- Bayesian Optimization function was more convenient to code
  - Subsequently used for ensemble models
- Voting and Stacking Ensemble models improved accuracy (slightly)
  - Would perform better if models were built on separate datasets

