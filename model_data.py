import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def knn_model(trainData, trainLabels,testData, testLabels ):
    all_scores = []
    # call in a loop
    for k in range(1, 30):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)
        score = model.score(testData, testLabels)
        all_scores.append(score)
    best_accuracy = max(all_scores)
    k = np.argmax(all_scores)

    # Now predict the values for the best k
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)
    predicted_values = model.predict(testData)

    return model, best_accuracy, k, predicted_values