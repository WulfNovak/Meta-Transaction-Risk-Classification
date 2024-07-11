# Meta-Transaction-Risk-Classification
The following notebook exists to achieve the following:
- Familiarize self with Python and scikit learn
- Use Halving Grid Search and Bayesian Optimization to select hyperparameters
- Practice coding Voting and Stacking Ensemble models in Python

Some conclusions from this notebook:
- Halving grid search is computationally efficient and effective
- Bayesian Optimization is less computationally efficient, but has potential to be more effective
- Prior experience with Voting and Stacking Ensemble models validates what this script shows;
  ensemble model inputs (Light GBM) have great sway over the model output, especially if developed on the same data.
  These may still be effictive if ensemble input models are developed on different partitions of the data. 
