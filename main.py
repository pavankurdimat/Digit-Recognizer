#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Project_Digit_Recognizer
#File Name: main.py
#Description: This project Identifies_Hand_Written_Digits
#----------------------------------------------------
#--------------------------------------------------------------------------------------
# Context
# The MNIST database (Modified National Institute of Standards and Technology database)
# is a large database of handwritten digits that is commonly used for training various
# image processing systems.The database is also widely used for training and testing
# in the field of machine learning.
#--------------------------------------------------------------------------------------
#-------Libraries---------#
#--------------------------------------------------------------------------------------
import pandas as pd
import load_data as ld
import explore_data as ed
import prepare_data as pd
import model_data as md
import validate_data as vd



#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    print ("\n\n------------------Loading the Dataset------------------")
    mnist = ld.read_data()
    print("\n------------------Loading Completed.....")

    # Display shape and size of the matrix
    print("\n\n------------------Display Shape and Size------------------")
    shape = ld.display_shape(mnist)
    print ("Shape of the Mnist Dataset:",shape )

    #Have a look at the data
    print("\n\n------------------Image Matrix------------------")
    n = int (input("Get to see the image matrix. Select the number in the range of 0 to {} : ".format(shape[0]) ))
    print (ld.get_image_matrix(mnist, n))

    print("\n\n------------------Plotting the Image------------------")
    ed.plot_image(mnist, n)
    print("\n------------------Get the Image Label------------------")
    print ("The label of the image is : ", ld.get_image_label(mnist, n))

    print ("\n\n------------------Prepare the Data------------------")
    trainData, testData, trainLabels, testLabels = pd.prepare_data(mnist)

    print("\n\n------------------Applying K Nearest Neighbors Model------------------")
    print ("Knn Model Started ....")
    model, best_accuracy, k, predicted_values = md.knn_model(trainData, trainLabels,testData, testLabels)
    print ("KNN Accuracy Values : ", best_accuracy)
    print("KNN Best k value : ", k)
    print("KNN model Completed ...")

    print("\n\n------------------Validate K Nearest Neighbors Model------------------")
    n = int(input("Test the KNN model. Select the row from the testData in the range of 0 to {} : ".format(shape[1])))
    pred_val , act_val = vd.validate_data(model, testData, testLabels, n)
    print ("The predicted Label is : ",pred_val)
    print ("The Actual Label is : ", act_val)

    vd.validation_report(predicted_values,testLabels)

    print("\n\n------------------Display errors K Nearest Neighbors Model------------------")
    vd.display_errors(predicted_values,testLabels)

    ## Assignment is Implement differenct kernels of Support Vector Machines model
    # and get the accuracy and then validate the model

    #pending Activity
    # https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

    print("\n\n------------------End of the Project------------------")
