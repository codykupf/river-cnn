#This file is used to process images

from whitebox import WhiteboxTools
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from shapely.geometry import Point,LineString, MultiPoint
from shapely.ops import split

#Cut out lakes from stream shapefile
def clip_lakes(site):

    # Initialize whitebox tools
    wbt = WhiteboxTools()

    # Sets the Whitebox working directory
    wbt.work_dir = "{}{}".format(folder, site)

    # Import the stream and watershed files
    s_file = "{}{}NHN_04HA000_2_0_HN_NLFLOW_1.shp".format(folder, site)
    wb_file = "{}{}NHN_04HA000_2_0_HD_WATERBODY_2.shp".format(folder, site)
    outfile = "{}{}s_erased.shp".format(folder, site)

    # Clip the streams to exclude lakes
    wbt.erase(s_file, wb_file, outfile)

#Filter shapefile segments
def filter_segments(site,s_clipped,epsg,min_length=500,min_vertices=100, max_vertices=224):

    print("Filtering from {} segments".format(s_clipped.shape[0]))

    # Drop un-necessary columns (keep NID and geometry)
    s_clipped.drop(s_clipped.columns.difference(['NID', 'geometry']), 1, inplace=True)

    # Reproject from EPSG:4269 to EPSG:2958 (NAD83(CSRS) / UTM zone 17N) so units are in metres
    s_clipped = s_clipped.to_crs(epsg)

    #Calculate the number of vertices per polyline segment
    vertices = []

    for i in range(0, s_clipped.shape[0]):
        vertices.append(len(s_clipped.loc[[i],'geometry'].values[0].xy[0]))
    s_clipped = s_clipped.assign(n_vertices=vertices)

    # Only keep segments with more than min number of vertices
    s_clipped = s_clipped[s_clipped['n_vertices'] > min_vertices].reset_index(drop=True)

    #Split long polylines at max_vertices (use floor of max)

    # Create a dataframe to hold the new split rows
    new_rows = gpd.GeoDataFrame()
    drop_rows = []

    for i in range(0,s_clipped.shape[0]):
        n_vertices = s_clipped.loc[[i],'n_vertices'].values

        if n_vertices > max_vertices:
            n_divs = math.ceil(n_vertices/max_vertices)

            #Get the points to split on
            points = [math.floor(n_vertices*j/n_divs) for j in range(n_divs+1)]
            #print("Splitting on ", points)

            #Split the polyline at the split points
            pline = s_clipped.loc[[i], 'geometry'].values[0]
            plines = split(pline, MultiPoint([pline.coords[j] for j in points[1:-1]]))

            #Create a new geodataframe row for each new line
            for line in plines:

                new_rows=new_rows.append(s_clipped.iloc[i],ignore_index=True)
                new_rows.loc[-1,'geometry'] = line

            #List of old rows to drop
            drop_rows.append(i)

    print("Dropping {} rows".format(len(drop_rows)))
    s_clipped = s_clipped.drop(drop_rows)

    print("Adding {} new rows".format(new_rows.shape[0]))
    s_clipped = s_clipped.append(new_rows,ignore_index=True).reset_index()

    '''
    for i in range(0, s_clipped.shape[0]):
        vertices.append(len(s_clipped.loc[[i],'geometry'].values[0].xy[0]))
    s_clipped = s_clipped.assign(n_vertices=vertices)
    '''
    # Calculate the length of each line
    s_clipped['length'] = s_clipped['geometry'].length

    # Only keep segements larger than minimum length
    s_clipped = s_clipped[s_clipped['length'] > min_length].reset_index()

    # Test plot
    #s_clipped.loc[[15], 'geometry'].plot()
    #plt.show()

    #Save clipped files
    s_clipped.to_file("{}s_filtered.shp".format(site))
    return(s_clipped)

#Function to convert polylines to images
def get_images(site,s_filtered):

    # Function to rotate points about the origin
    def rotate(x, y, theta):

        xx = x * math.cos(theta) + y * math.sin(theta)
        yy = -x * math.sin(theta) + y * math.cos(theta)
        return xx, yy

    def convert2img(pline, label, img_size=224):

        # Get a list of the X and Y coordinates
        x, y = pline[0].xy
        x = np.array(x)
        y = np.array(y)

        # If too few points discard
        if len(x) < 15:
            "Print discarding image due to too few points"
            return None

        else:

            # Translation to set first point equal to the origin 0,0
            x = x - x[0]
            y = y - y[0]
            angle = math.atan2(y[-1], x[-1])

            # Rotate the images to be horizontal
            x, y = rotate(x, y, angle)
            angle = math.atan2(y[-1], x[-1])

            # Remove bias from the image coordinates
            x = x - x.min()
            y = y - y.min()

            # Calculate the aspect ratio
            width_to_height = x.max() / y.max()

            # If ratio < 1 scale by y.max
            if width_to_height <= 1:
                x = (x / y.max() + (1 - width_to_height) / 2) * img_size
                y = y / y.max() * img_size
            # Otherwise scale by x.max
            else:
                y = (y / x.max() + ((1 - 1 / width_to_height) / 2)) * img_size
                x = x / x.max() * img_size

            # Create a list of pixel coordiantes
            pts = np.int32([x, y]).T

            canvas = np.ones((img_size, img_size, 3), np.uint8) * 255
            img = cv2.polylines(canvas, [pts], False, (0, 0, 0))

            # Return the image
            return img

    # Convert all lines to images
    count = 0
    saveto = "{}Images/".format(site)
    used = []

    print("Saving images to {}".format(saveto))

    # Loop through each image
    for i in range(0, s_filtered.shape[0]):

        line = s_filtered.loc[[i], 'geometry'].values
        img = convert2img(line, i)

        # If an image is returned
        if img is not None:
            used.append(i)

            # Save the image
            cv2.imwrite("{}{}.jpg".format(saveto, i), img)

            count = count + 1

    print("{} images exported".format(count))

if __name__ == '__main__':

    #Set folder and site
    folder = "/Users/codykupf/Documents/Projects/river-cnn-data/"
    site = "{}nhn_rhn_04ha000_shp_en/".format(folder)

    #clip surface waterbodies from the streams
    try:
        clipped = gpd.read_file("{}s_erased.shp".format(site))
    except:
        clipped = clip_lakes(site)

    #clipped.plot()
    #plt.show()

    #Reproject to UTM and filter
    #UTM 17N is EPSG:2958
    filtered = filter_segments(site,clipped,"EPSG:2958")

    filtered.plot()
    plt.show()

    #Normalize lines and convert to images
    get_images(site,filtered)




