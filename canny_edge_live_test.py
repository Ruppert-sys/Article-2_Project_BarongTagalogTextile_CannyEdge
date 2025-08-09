
import cv2
from datetime import datetime

def apply_canny_edge(frame_):
	gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edges_ = cv2.Canny(blurred, 25, 200)
	return edges_


cap = cv2.VideoCapture(1)

while True:
	ret, frame = cap.read()
	
	if not ret:
		break
	
	edges = apply_canny_edge(frame)

	cv2.imshow('Original', frame)
	cv2.imshow('Edges', edges)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	if cv2.waitKey(1) & 0xFF == ord('c'):
		td = datetime.now().strftime("%m%d%Y_%H%M%S.jpg")
		cv2.imwrite(td, edges)
		print('Captured!')

cap.release()
cv2.destroyAllWindows()
