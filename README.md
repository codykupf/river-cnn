# river-cnn

## Introduction
This repository contains code for the latest development version of the AI Tool that will be presented at the 2021 Canadian Society for Civil Engineering Hydrotechnical Conference in the paper **Development of an AI Tool to Identify Reference Reaches for Natural Channel Design** by Cody Kupferschmidt and Andrew Binns.

## Instructions for Use
This code is designed to be used as a tool for identifying *n* reference reaches within a distance *d* of a given site.
1. Download shapefiles for your study area watershed from the [National Hydrographic Network](https://open.canada.ca/data/en/dataset/a4b190fe-e090-4e6d-881e-b87956c07977).
2. Use the preprocess.py to pre-process the shapefile and generate .jpg files for each stream segment.
3. Extract features from each of the stream segment images using feature_extract.py
4. Use cluster.py to identify nearest neighbours for stream segments.


   
