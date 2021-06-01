import preprocess
#import featureextract
#import cluster
#import geopandas as gpd
from matplotlib import pyplot as plt
from os import path


# STEP 1. Preprocess
# This step will process an input shapefile and generate images

#Set working directory. It should contain five subfolders: Inputs, Results, Images, Models, Figures
folder = "" #local directory
inputs = "{}Inputs/".format(folder)
results = "{}Results/".format(folder)
images = "{}Images/".format(folder)
models = "{}Models/".format(folder)
figures = "{}Figures/".format(folder)


# List of input stream and watershed files
s_file = "{}NHN_04HA000_2_0_HN_NLFLOW_1.shp".format(inputs)
wb_file = "{}NHN_04HA000_2_0_HD_WATERBODY_2.shp".format(inputs)

# Specify local UTM Zone (UTM 17N is EPSG:2958)
utm_zone = "EPSG:2958"

#Set names for output files
clip_file = "{}s_clipped.shp".format(results)
filter_file = "{}s_filtered.shp".format(results)

#Load the file of the clipped waterbodies
try:
    print("Loading clipped file")
    clipped = gpd.read_file(clip_file)
    print("Clipped streams loaded from {}".format(clip_file))
#Or clip streams using waterbodies if the clipped file doesn't already exist
except:
    print("Clipping streams for {}".format(s_file))
    preprocess.clip_lakes(s_file,wb_file,clip_file)
    clipped = gpd.read_file(clip_file)

#Plot the clipped streams
clipped.plot()
plt.title("Clipped stream segments")
plt.show()

#Reproject to local UTM and filter
try:
    filtered = gpd.read_file(filter_file)
    print("Filtered streams loaded from {}".format(filter_file))
except:
    print("Filtering streams for {}".format(s_file))
    #Default optional arguments (min_length = 500, min_vertices = 100, max_vertices = 224)
    filtered = preprocess.filter_segments(clip_file,filter_file,utm_zone)

#Plot the filtered streams
filtered.plot()
plt.title("Filtered stream segments")
plt.show()



#Extract images from the filtered file
preprocess.create_images(filter_file,images)


#STEP 2. Extract features using VGG16 and ResNet50

# Run feature extraction for files that don't exist
if path.exists("{}VGG_Results.hdf5".format(models)) is False:
    featureextract.run_vgg(images, models)

if path.exists("{}ResNet_Results.hdf5".format(models)) is False:
    featureextract.run_resnet(images, models)



#STEP 3. Cluster

#Flatten each of the models
vgg = cluster.flatten("{}VGG_Results.hdf5".format(models), "VGG")
resnet = cluster.flatten("{}ResNet_Results.hdf5".format(models), "ResNet")

#Calculate the distance matrices
distance_vgg = cluster.solve_distance(vgg)
distance_resnet = cluster.solve_distance(resnet)

#Find the closest
outfile_vgg = '{}.png'.format("VGG_Closest")
cluster.plot_closest(distance_vgg, 6, images, outfile_vgg, model="knn")

outfile_resnet = '{}.png'.format("ResNet_Closest")
cluster.plot_closest(distance_resnet, 6, images, outfile_resnet, model="knn")

