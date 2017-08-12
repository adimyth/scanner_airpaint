from collections import deque
import numpy as np
import imutils
import cv2

# creating a canvas to save the image
canvas = np.ones((512,512,3),np.uint8)
# threshold range for R,G,B
green_lower = (50,100,100)
green_upper = (70,255,255)
blue_lower  = (110,100,100)
blue_upper  = (130,255,255)
# HSV is cylindrical co-ordinate system hence there are 2 ranges for red color
red_lower1	= (0,70,50)
red_upper1	= (10,255,255)
red_lower2	= (170,70,50)
red_upper2 	= (180,255,255)
# buffer to store the tracing path
pts	= deque(maxlen = 30)
arr = []
# create an object to use webcam
camera = cv2.VideoCapture(0)
while(True and camera.isOpened()):
	# grab the frame
	grabbed,frame = camera.read()

	# resize,blur and convert the frame to HSV
	frame = imutils.resize(frame,width=600)
	blur  = cv2.GaussianBlur(frame,(11,11),0)
	hsv   = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

	# create a mask for identifying the green color
	mask = cv2.inRange(hsv,blue_lower,blue_upper)
	# kernel = None ;because, I'm not too sure abt it and the OpenCV reference wasn't convincing either
	mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,None)
	# removing noise(Erosion followed by Dilation)

	# find the contours of the objects in the frame
	cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	# check if atleast one contour is present
	if len(cnts) > 0:
		# for every contour obtained in the frame
		for cnt in cnts:
			approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
			# check if there is a circle; for other shapes check the number of vertices
			if len(approx)>8:
				# add them to a new list
				arr.append(cnt)
		
		# find out the largest circular shape with selected color
		c = max(cnts,key = cv2.contourArea)
		# finding the center & radius of the circle
		((x,y),radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		# finding the centroid
		center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
		if radius > 1:
			# for the boundary around the surface
			cv2.circle(frame,(int(x),int(y)),int(radius),(255,0,0),3)
			# for the centroid
			cv2.circle(frame,center,5,(0,0,255),-1)
			# tracing lines
		pts.appendleft(center)

	for i in range(1,len(pts)):
		if pts[i-1] is None or pts[i] is None:
			continue
		thickness = int(np.sqrt(30/float(i+1)) * 2.5)
		cv2.line(frame,pts[i-1],pts[i],(0,0,255),thickness)
		cv2.line(canvas,pts[i-1],pts[i],(0,0,255),thickness)
	cv2.imshow("Object Tracking",frame)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break
camera.release()
cv2.imwrite('Paint.jpg',canvas)
cv2.destroyAllWindows() 