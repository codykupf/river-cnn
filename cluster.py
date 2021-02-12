import numpy as np
import h5py
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.spatial import distance_matrix


def flatten(model_file,model):

    if model == "ResNet":
        shape = 7*7*2048
    else:
        shape = 7*7*512

    # Load resnet results
    with h5py.File(model_file, "r") as f:
        # Get the data
        model_data = f.get('Results')[()]

    length = model_data.shape[0]
    print("Length is", model_data.shape[0])
    print("Width is",shape)

    # Flatten model data
    model_data = np.array(model_data).flatten()
    model_data = model_data.reshape(length,shape)
    print(model_data.shape)

    return model_data

def get_clusters(model,K):

    #Identify
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(model)
    labels = kmeans.predict(model)
    distances = kmeans.transform(model) ** 2
    df = pd.DataFrame(distances.sum(axis=1), columns=['sq-dist'])
    df['label'] = labels

    # Sort
    df = df.sort_values(['label', 'sq-dist'])
    plot_closest(df, k, images)

def plot_closest(df,K,images,n_figs=5,model="kmeans"):

    fig = plt.figure()
    # fig.suptitle('{}'.format(label), fontsize=16)

    # Loop through each of the K terms and find the n closest images
    for row in range(0, K):

        if model == "kmeans":

            #Get the n closest images
            subset = df[df['label']==row].head(n_figs)

            plot_images = subset.index.values

        #Othrew
        else:

            #Pick a random row
            rand_row = np.random.randint(0,df.shape[0])
            plot_images = np.argsort(distances[:,rand_row])[:n_figs]

        #Load and plot the images
        for i in range(0, n_figs):

            image = mpimg.imread("{}{}.jpg".format(images,plot_images[i]))
            #print("Loading {}{}.jpg".format(images,plot_images[i]))

            ax = fig.add_subplot(8, n_figs, 1 + i + row * n_figs)

            # Thicken the line for plotting to 2 px
            kernel = np.ones((2, 2), np.uint8)
            thickened = cv2.erode(image, kernel, iterations=1)
            ax.imshow(thickened, aspect='equal')

            # ax.imshow(image, aspect='equal')
            # plt.axis('off')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.spines['bottom'].set_color(None)
            ax.spines['top'].set_color(None)
            ax.spines['right'].set_color(None)
            ax.spines['left'].set_color(None)

            if i == 0:
                ax.set_ylabel("Class {}".format(row + 1), size='large')

    plt.subplots_adjust(top=1.0, wspace=0.1, hspace=0.05)
    plt.show()

if __name__ == '__main__':

    #Set folder and site
    folder = "/Users/codykupf/Documents/Projects/river-cnn/"
    images = "{}Images/".format(folder)
    models = "{}Models/".format(folder)

    # Flatten model results for clustering
    print("Flattening model results")
    vgg = flatten("{}VGG_Results.hdf5".format(models), "VGG")
    resnet = flatten("{}ResNet_Results.hdf5".format(models), "ResNet")

    i = 1

    model = vgg

    # Drop features that are the same for all points
    model = model[:, ~np.all(model == model[0, :], axis=0)]
    print("Input shape is", model.shape)

    '''
    # Perform k-means++ clustering
    for k in [2]:
        print("Performing k-means clustering for VGG k={}".format(k))
        get_clusters(model,k)
    '''

    #Calculate the distance matrix (this is slow)
    print("Calculating distance matrix")
    distances = distance_matrix(model,model,2)

    #Find kNNs for n given points
    print("Plotting kNN")
    plot_closest(distances,6,images,model="knn")







