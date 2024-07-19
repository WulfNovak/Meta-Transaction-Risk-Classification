# train / test split with class imbalance using downsampling
import pandas as pd
from sklearn.model_selection import train_test_split

# target_variable: name of target variable
# testset_size: the percentage of data that should be in the test set

def class_imbalance_train_test_split(data, target_variable, testset_size):
       
       # Training Set and validation set
       df_x = data.drop(columns = ['anomaly', 'anomaly_binary'])
       df_y = data[target_variable] # as category?
       
       ### Train / Test split. Stratify by underrepresented class
       x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = testset_size, stratify = df_y, random_state = 152)

       # Recombine Training sets to downsample
       training_sets = pd.concat([x_train, y_train], axis = 1)
       min_class_size = training_sets[target_variable].value_counts().min() # underepresented class (high risk) count
       print('Class Imbalance')
       print(training_sets[target_variable].value_counts())

       # Downsample training set to resolve problems with imbalanced data
       downsampled_train = (training_sets.groupby(target_variable, group_keys = False)
                            .apply(lambda x: 
                                   x.sample(n = min_class_size, random_state = 44, replace = True))
                            # Reshuffle 
                            .sample(frac = 1, random_state = 42))

       # Override training sets with downsampled versions
       x_train = downsampled_train.drop(columns = target_variable)
       y_train = downsampled_train[target_variable]

       print("\n")
       print('Training Set Class Balance')
       print(y_train.value_counts())

       print("\n")
       print('Validation Set Class Balance ')
       print(y_test.value_counts())
       
       return x_train, x_test, y_train, y_test
