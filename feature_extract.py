import glob
import numpy as np
import h5py
from os import path

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


