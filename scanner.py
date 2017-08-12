import imutils
import cv2
import numpy as np
from skimage.filters import threshold_adaptive
import argparse

def order_pts(pts):
	rect = np.zeros((4,2),dtype="float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]		#top-left point has the smallest x+y
	rect[2] = pts[np.argmax(s)]		#bottom-right point has the largest x+y
	diff = np.diff(pts,axis=1)
	rect[1] = pts[np.argmin(diff)]	#top-right point has the smallest x-y
	rect[3] = pts[np.argmax(diff)]	#bottom-left point has the largest x-y
	return rect

# pts is the list of points that contain the ROI to be transformed
def four_point_transform(image,pts):
	rect = order_pts(pts)
	(tl,tr,bl,br) = rect
	widthA   = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB   = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxwidth = max(widthA,widthB)
	heightA  = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB  = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - tl[1]) ** 2))
	maxheight= max(heightA,heightB)

	dst = np.array([[0,0],[maxwidth-1,0],[maxwidth-1,maxheight-1],[0,maxheight-1]],dtype="float32")
	M = cv2.getPerspectiveTransform(rect,dst)
	warped = cv2.warpPerspective(image,M,(int(maxwidth),int(maxheight)))
	return warped

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
copy = image.copy()
ratio = image.shape[0]/500.0
image = imutils.resize(image,height=500)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
# image,threshold1,threshold2
edged = cv2.Canny(gray,75,200)
# Find all the contours in the image & sort them according to their area in decreasing order
im2, cnts, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)
# Traverse all the contours & find the one which is largest rectangle 
for c in cnts:
	perimeter = cv2.arcLength(c,True)	# (curve,closed)
	approx = cv2.approxPolyDP(c,0.02*perimeter,True)	#tolerance or closeness
	if len(approx) == 4:
		screenCnt = approx
		break
# Obtain Bird-Eye View
warped = four_point_transform(copy,screenCnt.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped,block_size = 251,offset = 10,method = 'gaussian')
warped = warped.astype("uint8") * 255

cv2.imshow("Image",copy)
cv2.imshow("Edged",edged)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.imshow("Original", imutils.resize(copy, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)