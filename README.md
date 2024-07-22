<p align="center"> # Meta-Transaction-Risk-Classification</p>

### **Notebook exists to achieve the following:**
- Explore Meta Transaction Data
   - Engineer Features
- Utilize Halving Grid Search and Bayesian Optimization
   - Will only use on Light GBM
   - Hyperparameter optimization
   - Compare performance metrics on validation set
   - Compare ease of coding 
- Using scikit-learn, create model pipeline to test different model types.
- Combine models to voting and stacking ensemble models, compare resulting accuracy.

### **Order to View or Run Scripts:**
1. meta_eda
2. meta_ml
   - This includes functions given in the functions folder
3. meta_ensemble_pipeline

### **Noted Results:**
- Deduced that data is synthetic
- Multicollinearity between several key variables and target variable, anomaly
  - Multicollinear variables: risk_score, transaction_type
  - Removed variables from analysis
- Halving Grid Search resulted in *marginally* better Light GBM models for this objective
- Bayesian Optimization function was more convenient to code
  - Subsequently used for ensemble models
  - less run time
- Voting and Stacking Ensemble models improved accuracy (slightly)
  - Would perform better if models were built on separate datasets

