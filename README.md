<h1 align="center"> Meta-Transaction-Risk-Classification</h1>

### **Notebook exists to achieve the following:**
- Analyze and Prepare Meta Transaction data for analysis
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
3. meta_ensemble_pipeline

### **Noted Results:**
- Deduced that data is synthetic
- Multicollinearity between several key variables and target variable, anomaly
  - Multicollinear variables: risk_score, transaction_type
  - Removed variables from analysis
- Bayesian Optimization resulted in *marginally* better Light GBM models for this objective
   - Bayesian Optimization found superior hyperparameters more quickly
   - Resulting models required fewer input variables
   - Used for ML Pipeline and ensemble model section
  
- Voting and Stacking Ensemble models did not outperform ensemble inputs
  - May perform better if models were built on separate datasets

