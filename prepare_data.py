from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(mnist):
    print ("Data Preparation Started....")
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target,
                                                                      test_size=0.25, random_state=42)
    print("Shape of the Features of the Training Dataset : ",trainData.shape)
    print("Shape of the Labels of the Training Dataset : ", trainLabels.shape)
    print("Shape of the Features of the Testing Dataset : ", testData.shape)
    print("Shape of the Labels of the Testing Dataset : ", testLabels.shape)

    return (trainData, testData, trainLabels, testLabels)

