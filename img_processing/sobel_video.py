import cv2
import numpy as no

from sobel import sobel
from imutils.video import WebcamVideoStream

vs = WebcamVideoStream(src = 0).start()

while(True):
	frame = vs.read()

	magnitude, theta = sobel(frame)
	print()

	cv2.imshow("Normal", frame)
	cv2.imshow("Sobel", magnitude)

	key = cv2.waitKey(0)
	if(key == ord("q")):
		break

vs.stop()
cv2.destroyAllWindows()