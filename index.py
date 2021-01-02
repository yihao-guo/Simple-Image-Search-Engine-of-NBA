# USAGE
# python index.py --dataset images --index index.cpickle

# import the necessary packages
from pyimagesearch.rgbhistogram import RGBHistogram
from imutils.paths import list_images
import argparse
import _pickle as cPickle     #这里改了
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])

# use list_images to grab the image paths and loop over them
for imagePath in list_images(args["dataset"]):
	# extract our unique image ID (i.e. the filename)
	imagePath = imagePath.replace("\\","/")
	k = imagePath[imagePath.rfind("/") + 1:]
	#这里改了
	#j, k = os.path.split(imagePath)

	# load the image, describe it using our RGB histogram
	# descriptor, and update the index
	image = cv2.imread(imagePath)
	features = desc.describe(image)
	index[k] = features

# we are now done indexing our image -- now we can write our
# index to disk
f = open(args["index"], "wb")
f.write(cPickle.dumps(index))  #改
f.close()

# show how many images we indexed
print("[INFO] done...indexed {} images".format(len(index)))
