# Create feature importance plot given model predictions for light GBM model
from lightgbm import LGBMClassifier
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def LGBM_feature_importance(model_output, x_test, y_test):

    # Feature Importance Plot
    opt_model = LGBMClassifier(**model_output.best_params_).fit(x_test, y_test)

    feature_importance = opt_model.feature_importances_
    feature_names = x_test.columns.to_list()

    importances_df = pd.DataFrame({"feature_names" : feature_names, 
                                "importances" : feature_importance})

    feature_importance_plot = sb.barplot(data = importances_df,
                                        x = feature_importance,
                                        y = feature_names,
                                        order = importances_df.sort_values('importances', ascending = False).feature_names)

    plt.title('Feature Importance Plot on Training Set')
    
    return feature_importance_plot