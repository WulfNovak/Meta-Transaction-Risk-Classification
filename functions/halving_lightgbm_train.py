# halving grid search for light gbm
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# objective can be 'binary' or 'multiclass'

def halving_lightgbm_train(objective, x_train, y_train): 
   # HalvingGridSearch with Light GBM
   classifier = LGBMClassifier()
   
   param_grid = {'objective': [objective], 
               'learning_rate': [.05, .1, .2, .25, .4, .55, .75, 2], 
               'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 
               'feature_fraction': [.01, .1, .2, .3, .4, .5],
               'subsample': [.01, .1, .2, .3, .4, .5],
               'num_leaves': [2**4, 2**5, 2**6, 2**7], 
               'verbosity': [-1],
               'force_col_wise': [True]
               }


   # HalvingGridSearchCV
   halving_grid_search = (HalvingGridSearchCV(classifier, param_grid, cv = 2, random_state = 41, n_jobs = -1) 
                     .fit(x_train, y_train)); 

   # print(f"Best parameters for {classifier}:\n{halving_grid_search.best_params_}")
   # print(f"Best score: {halving_grid_search.best_score_:.4f}\n")

   return halving_grid_search
