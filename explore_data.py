#--------------------------------------------------------------------------------------
#-------Explore Data---------#
#--------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_image(mnist, n):
    plt.figure(figsize = (5,5))
    plt.matshow(mnist.images[n])
    plt.show()
    return 0
