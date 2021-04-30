import preprocess
import geopandas as gpd
from matplotlib import pyplot as plt

# STEP 1. Preprocess
# This step will process an input shapefile and generate images

#Set working directory. We use three subfolders: Inputs, Results, and Images
folder = "/Users/codykupf/Documents/Projects/river-cnn/"
inputs = "{}Inputs/".format(folder)
results = "{}Results/".format(folder)
images = "{}Images/".format(folder)

'''

# List of input stream and watershed files
s_file = "{}NHN_04HA000_2_0_HN_NLFLOW_1.shp".format(inputs)
wb_file = "{}NHN_04HA000_2_0_HD_WATERBODY_2.shp".format(inputs)

#Set names for output files
clip_file = "{}s_clipped.shp".format(results)
filter_file = "{}s_filtered.shp".format(results)

#clip surface waterbodies from the streams if the clipped file doesn't already exist
try:
    clipped = gpd.read_file(clip_file)
    print("Clipped streams loaded from {}".format(clip_file))
except:
    print("Clipping streams for {}".format(s_file))
    preprocess.clip_lakes(s_file,wb_file,clip_file)
    clipped = gpd.read_file(clip_file)

#Plot the clipped streams
clipped.plot()
plt.title("Clipped stream segments")
plt.show()

#Reproject to UTM and filter
#UTM 17N is EPSG:2958
try:
    filtered = gpd.read_file(filter_file)
    print("Filtered streams loaded from {}".format(filter_file))
except:
    print("Filtering streams for {}".format(s_file))
    filtered = preprocess.filter_segments(clip_file,filter_file,"EPSG:2958")

#Plot the filtered streams
filtered.plot()
plt.title("Filtered stream segments")
plt.show()
'''