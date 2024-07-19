# Confusion Matrix Dependent on binary or multi-class output 

# This function does not generalize to other datasets

import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Objective is'binary' or 'multiclass'
def meta_confusion_matrix(y_pred, y_test, objective): 
    # Report and Confusion Matrix
    print(classification_report(y_test, y_pred))

    print('\n')

    if objective == 'binary': # Objective is define globally ('binary' or 'multiclass')
        class_labels = [0, 1]   

    else:    
        class_labels = ['low_risk', 'moderate_risk', 'high_risk']
    
    cm = confusion_matrix(y_test, y_pred, labels = class_labels) # Are these labels in the right order? 

    sb.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", xticklabels = class_labels, yticklabels = class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show(); 