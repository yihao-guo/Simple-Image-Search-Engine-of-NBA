# import the necessary packages
#加入关键的包
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.searcher import Searcher
import numpy as np
import argparse
import os
import _pickle as cPickle
import cv2

# construct the argument parser and parse the arguments
#增加输入的格式
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where we stored our index")
ap.add_argument("-q", "--query", required = True,
	help = "Path to query image")
args = vars(ap.parse_args())

# load the query image and show it
#显示要查询的图片
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", queryImage)
print("query: {}".format(args["query"]))

# describe the query in the same way that we did in
# index.py -- a 3D RGB histogram with 8 bins per
# channel
#建立直方图
desc = RGBHistogram([8, 8, 8])
queryFeatures = desc.describe(queryImage)

# load the index perform the search
index = cPickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)
results = searcher.search(queryFeatures)

# initialize the two montages to display our results --
# we have a total of 25 images in the index, but let's only
# display the top 10 results; 5 images per montage, with
# images that are 400x166 pixels
#建立两个图片框用来显示要展示的图片
montageA = np.zeros((166 * 3, 400, 3), dtype = "uint8")
montageB = np.zeros((166 * 3, 400, 3), dtype = "uint8")

# loop over the top ten results
for j in range(0, 6):
	# grab the result (we are using row-major order) and
	# load the result image
	#
	(score, imageName) = results[j]
	path = os.path.join(args["dataset"], imageName)
	result = cv2.imread(path)
	print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
	#输出：\t表示表格格式   {}表示要显示的位置  {：.3f}表示结果保留三位小数 ,format里的三个值分别对应三个{}里的值。

	# check to see if the first montage should be used
	if j < 3:
		montageA[j * 166:(j + 1) * 166, :] = result

	# otherwise, the second montage should be used
	else:
		montageB[(j - 3) * 166:((j - 3) + 1) * 166, :] = result

# show the results
#显示结果
cv2.imshow("Results 1", montageA)
cv2.imshow("Results 2", montageB)
cv2.waitKey(0)
