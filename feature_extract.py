import glob
import numpy as np
import h5py
import os.path
import pandas as pd
from os import path
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2

#Keras libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#Function to extract features for various models
def extract(images, model):

  #Get a list of all the input images
  img_paths = glob.glob("{}*.jpg".format(images))

  results = []
  counter = 1

  #Pass each image through the CNN to reduce the dimensionality
  for img_path in img_paths:

    print("Processing img {} of {}".format(counter,len(img_paths)))

    img = image.load_img(img_path,target_size=(224,224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    results.append(model.predict(img_data))

    counter = counter + 1

  results_array = np.array(results)
  print(results_array.shape)

  return results_array

def run_vgg(images,models):

    print("Running VGG")

    # Initialize the model using the weights for imagenet
    vgg_model = VGG16(weights='imagenet', include_top=False)

    # Display VGG model settings
    # vgg_model.summary()

    # Run VGG16 Model to extract features
    vgg_results = extract(images, vgg_model)

    # Export the results to an HDF5 file
    hfile = "{}VGG_Results.hdf5".format(models)

    with h5py.File(hfile, 'w') as f:
        dset = f.create_dataset("Results", data=vgg_results)

def run_resnet(images,models):

    print("Running ResNet")

    # Initialize the model using the weights for imagenet
    model_resnet = ResNet50(weights='imagenet', include_top=False)

    # Show model params
    # model_resnet.summary()

    # Run model to extract features
    resnet_results = extract(images, model_resnet)

    # Export results to hdf
    hfile = "{}ResNet_Results.hdf5".format(models)

    with h5py.File(hfile, 'w') as f:
        dset = f.create_dataset("Results", data=resnet_results)

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

def plot_closest(df,K,images,n_figs=8):

    fig = plt.figure()
    # fig.suptitle('{}'.format(label), fontsize=16)

    # Loop through each of the K terms and find the n closest images
    for row in range(0, K):

        #Get the n closest images
        subset = df[df['label']==row].head(n_figs)

        print(subset)

        plot_images = subset.index.values

        print(plot_images)

        for i in range(0, n_figs):

            image = mpimg.imread("{}{}.jpg".format(images,plot_images[i]))
            print("Loading {}{}.jpg".format(images,plot_images[i]))

            ax = fig.add_subplot(4, n_figs, 1 + i + row * n_figs)

            # Thicken the line
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

    #Run if files don't exist
    if path.exists("{}VGG_Results.hdf5".format(models)) is False:
        run_vgg(images,models)
    if path.exists("{}ResNet_Results.hdf5".format(models)) is False:
        run_vgg(images,models)

    #Flatten model results for clustering
    vgg = flatten("{}VGG_Results.hdf5".format(models),"VGG")
    resnet = flatten("{}ResNet_Results.hdf5".format(models),"ResNet")

    i = 1

    #Perform clustering
    for k in [4]:

        print("Performing k-means clustering for {}".format(k))
        kmeans = KMeans(n_clusters=k,init='k-means++')
        kmeans.fit(resnet)
        labels = kmeans.predict(resnet)
        distances = kmeans.transform(resnet)**2
        df = pd.DataFrame(distances.sum(axis=1),columns=['sq-dist'])
        df['label']=labels

        #Sort
        df = df.sort_values(['label','sq-dist'])
        plot_closest(df,k,images)



