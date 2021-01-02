# import the necessary packages
#加入关键的包
from pyimagesearch.searcher import Searcher
import numpy as np
import argparse
import os
import _pickle as cPickle
import cv2

# construct the argument parser and parse the arguments
#创建输入时的格式要求
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where we stored our index")
args = vars(ap.parse_args())

# load the index and initialize our searcher
#从磁盘读取图片的index
index = cPickle.loads(open(args["index"], "rb").read())
#初始化searcher类的对象
searcher = Searcher(index)

# loop over images in the index -- we will use each one as
# a query image
for (query, queryFeatures) in index.items():
	# perform the search using the current query
	results = searcher.search(queryFeatures)

	# load the query image and display it
	#加载并显示查询的图像
	path = os.path.join(args["dataset"], query)
	queryImage = cv2.imread(path)
	cv2.imshow("Query", queryImage)
	print("query: {}".format(query))

	# initialize the two montages to display our results --
	# we have a total of 25 images in the index, but let's only
	# display the top 10 results; 5 images per montage, with
	# images that are 400x166 pixels
	#定义整个蒙太奇图片的大小
	montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
	montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")

	# loop over the top ten results
	#把前十相似度的图片分别用蒙太奇图片的形式展现出来；
	for j in range(0, 10):
		# grab the result (we are using row-major order) and
		# load the result image
		(score, imageName) = results[j]
		path = os.path.join(args["dataset"], imageName)
		result = cv2.imread(path)
		print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))

		# check to see if the first montage should be used
		if j < 5:
			montageA[j * 166:(j + 1) * 166, :] = result

		# otherwise, the second montage should be used
		else:
			montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

	# show the results
	#展示蒙太奇图片
	cv2.imshow("Results 1-5", montageA)
	cv2.imshow("Results 6-10", montageB)
	cv2.waitKey(0)