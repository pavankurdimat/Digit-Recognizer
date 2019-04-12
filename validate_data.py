import pandas as pd
from sklearn import metrics
import seaborn as sns

def validate_data(model, testData, testLabels , n):
    pred_val = model.predict(pd.DataFrame(testData[n]).T)
    act_val = testLabels[n]
    return pred_val, act_val


def validation_report(predicted_values,testLabels):
    print ("Confusion Matrix")
    print ("=================")
    cm = metrics.confusion_matrix(predicted_values,testLabels)
    print (cm)
    sns.heatmap(cm, annot = True)

    # plt.imshow(cm, interpolation='nearest')
    # plt.title("Confusion Matrix Plot")
    # plt.colorbar()
    # classes = range(10)
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # plt.show()

def display_errors(predicted_values,testLabels):
    error_boolean_array = predicted_values != testLabels

    pred = predicted_values[error_boolean_array]
    act = testLabels[error_boolean_array]
    tot_errors = sum(error_boolean_array)

    print ("Total Labels that are tested :", len(testLabels))
    print ("Out of {} total {} are correctly predicted and {} are wrongly predicted ".format(len(testLabels),
                                                                                             len(testLabels) - tot_errors ,
                                                                                             tot_errors))

    for val in range(tot_errors):
        print ("Actual Label : {} Predicted Label : {} ".format( act[val], pred[val]))




