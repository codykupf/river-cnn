import numpy as np
import h5py
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA



def flatten(model_file,model):

    if model == "ResNet":
        shape = 7*7*2048
    elif model == "VGG":
        shape = 7*7*512
    elif model == "Fine Tune":
        shape = 512
    else:
        print('Invalid model name')

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

def get_clusters(model,K,images):

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

def plot_closest(df,K,images,outfile,n_figs=8,model="kmeans"):

    np.random.seed(36)
    fig = plt.figure(figsize=(4,4), dpi=300)
    # fig.suptitle('{}'.format(label), fontsize=16)

    # Loop through each of the K terms and find the n closest images
    for row in range(0, K):

        if model == "kmeans":

            print(df[df['label']==row])
            #Get the n closest images
            subset = df[df['label']==row].head(n_figs)

            plot_images = subset.index.values
            print(plot_images)

        #Otherwise Knn
        else:

            #Pick a random row
            rand_row = np.random.randint(0,df.shape[0])
            print("Random row is ", rand_row)
            plot_images = (np.argsort(distances[:,rand_row])[:n_figs])
            print(plot_images)

            #Occasionally some points are duplicate distances of zeroes
            #Drop the rand_row from plot_images and append rand_row to front
            plot_images = np.delete(plot_images,np.where(plot_images==rand_row))
            plot_images = np.insert(plot_images,0,rand_row)
            print(plot_images)

        #Load and plot the images
        for i in range(0, n_figs):

            image = mpimg.imread("{}{}.jpg".format(images,plot_images[i]))
            print("Loading {}{}.jpg".format(images,plot_images[i]))

            ax = fig.add_subplot(6, n_figs, 1 + i + row * n_figs)

            # Thicken the line for plotting to 2 px
            kernel = np.ones((2, 2), np.uint8)
            thickened = cv2.erode(image, kernel, iterations=1)
            ax.imshow(thickened, aspect='equal')

            if i == 0:
                title_text = 'SS{}'.format(plot_images[i])
            else:
                title_text = 'RR{}'.format(plot_images[i])

            ax.set_title(title_text,fontsize=6,pad=-40)

            # ax.imshow(image, aspect='equal')
            # plt.axis('off')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.spines['bottom'].set_color(None)
            ax.spines['top'].set_color(None)
            ax.spines['right'].set_color(None)
            ax.spines['left'].set_color(None)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(outfile)
    plt.show()

if __name__ == '__main__':

    #Set folder and site
    folder = "/Users/codykupf/Documents/Projects/river-cnn/"
    images = "{}Images/".format(folder)
    models = "{}Models/".format(folder)
    figures = "{}Figures/".format(folder)

    # Flatten model results for clustering
    print("Flattening model results")
    vgg = flatten("{}VGG_Results.hdf5".format(models), "VGG")
    resnet = flatten("{}ResNet_Results.hdf5".format(models), "ResNet")
    vgg_finetune = flatten("{}VGG_FineTune1e-4_Results.hdf5".format(models), "Fine Tune")
    resnet_finetune = flatten("{}ResNet_FineTuneA_Results.hdf5".format(models), "Fine Tune")

    all_models= [vgg, resnet, vgg_finetune, resnet_finetune]
    labels = ["VGG16", "ResNet50","VGG16-FineTune","ResNet50-FineTune"]

    for i in range(0,len(all_models)):

        print("Processing {}".format(labels[i]))

        model = all_models[i]

        #Perform PCA
        print("Performing PCA")
        pca = PCA(n_components=100,svd_solver='full')
        model = pca.fit_transform(model)

        #Drop rows not within distance d

        #Calculate the distance matrix (this is slow)
        print("Calculating distance matrix")
        distances = distance_matrix(model,model,2)
        #distances = distance_matrix(model,model,2)

        #Find kNNs for n given points
        print("Plotting kNN")
        #between 5 and 7 seems to be the best so far
        outfile = '{}{}.png'.format(figures,labels[i])
        plot_closest(distances,6,images,outfile,model="knn")








