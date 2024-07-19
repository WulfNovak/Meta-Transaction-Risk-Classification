from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

def bayesian_lightgbm_train(objective, x_train, y_train): # 'binary' or 'multiclass'
    # Model Classifier
    classifier = LGBMClassifier()
    
    # Parameter Space
    param_space = {
    'objective': [objective],
    'learning_rate': Real(0.01, 1, prior = 'uniform'),
    'num_leaves': Integer(10, 50),
    'max_depth': Integer(3, 10),
    'boosting_type': Categorical(['gbdt', 'dart']),
    'feature_fraction': Real(0.1, 1, prior = 'uniform'),
    'subsample': Real(0.1, 1, prior = 'uniform'),
    'verbosity': [-1],
    'force_col_wise': [True]
    }

    # Bayesian Optimization with Cross Validation
    light_gbm_bayes_cv = BayesSearchCV(
        estimator = LGBMClassifier(),
        search_spaces = param_space,
        scoring = 'accuracy', 
        cv = 5,
        n_iter = 50, 
        n_jobs = -1,
        return_train_score = True,
        random_state = 42 
    ).fit(x_train, y_train); 

    # print('\n')
    # print(f"Best parameters for {classifier}:\n{light_gbm_bayes_cv.best_params_}")
    # print("Best Score:", light_gbm_bayes_cv.best_score_)
    
    return light_gbm_bayes_cv