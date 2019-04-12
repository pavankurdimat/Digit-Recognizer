#--------------------------------------------------------------------------------------
#-------Load Data---------#
#--------------------------------------------------------------------------------------
from sklearn import datasets
def read_data():
    mnist = datasets.load_digits()
    return mnist

def display_shape(mnist):
    shape = mnist.data.shape
    return shape

def get_image_matrix(mnist, n):
    image_matrix = mnist.images[n]
    return image_matrix

def get_image_label(mnist, n):
    return mnist.target[n]