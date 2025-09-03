
import cv2
import time
import platform
import numpy as np

# tensorFlow is required for Keras to work
from keras.models import load_model

# disable scientific notation for clarity
np.set_printoptions(suppress=True)

from multiprocessing import Process, Queue

from src.constants import *

if RPI_PLATFORM:
	from picamera2 import Picamera2


class ServiceGrabber:
	
	def __init__(self):
		
		self.tx = Queue()
		self.rx = Queue()
		
		self.grabber_process = Process(target=self.grab_frames, args=(self.tx, self.rx))
		self.grabber_process.start()
	
	@staticmethod
	def grab_frames(rx, tx):
		
		model = load_model(MEAT_MODEL, compile=False)
		class_names = open(MEAT_LABELS, "r").readlines()
		
		if RPI_PLATFORM:
			camera = Picamera2()
			
			preview_config = camera.create_preview_configuration(main={"size": IMG_SIZE})
			camera.configure(preview_config)
			
			camera.start()
			time.sleep(2)
		else:
			camera = cv2.VideoCapture(DEFAULT_CAMERA_IDX)
			camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		
		data = None
		
		print('[INFO] Camera service started!')
		try:
			while True:
				
				if not rx.empty():
					data = rx.get()
					# print(data)
				
				if data is not None:
					
					if data[0] == QUIT:
						if RPI_PLATFORM:
							camera.close()
						else:
							camera.release()
						break
					
					elif data[0] == CAPTURE_IMG:
						
						if RPI_PLATFORM:
							# take a picture
							camera.capture_file(TEMP_IMG)
							time.sleep(.1)
							camera_frame = cv2.imread(TEMP_IMG, cv2.IMREAD_GRAYSCALE)
							camera_frame = cv2.Canny(camera_frame, 100, 200)
							cv2.imwrite(TEMP_IMG, camera_frame)

						else:
							has_frame, camera_frame = camera.read()
							if has_frame:
								_, camera_frame = camera.read()

								camera_frame = cv2.Canny(camera_frame, 100, 200)
								
								cv2.imwrite(TEMP_IMG, camera_frame)
								time.sleep(.1)
						
						print("Image has been captured!")
						# send a command to the main thread done taking a picture
						tx.put((CAPTURE_IMG_DONE, None))
					
					elif data[0] == CLASSIFY_IMG:
						
						# load the captured image
						image = cv2.imread(TEMP_IMG)
						
						# resize the raw image into (224-height,224-width) pixels
						image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
						
						# make the image a numpy array and reshape it to the models input shape.
						image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
						
						# normalize the image array
						image = (image / 127.5) - 1
						
						# predicts the model
						prediction = model.predict(image)
						index = np.argmax(prediction)
						class_name = class_names[index]
						class_name = class_name[2:].strip()
						
						confidence_score = prediction[0][index]
						#confidence_score = str(np.round(confidence_score * 100))[:-2] 
						
						prediction_details = {
							"class_name": class_name,
							"confidence_score": confidence_score
						}
						
						# send the prediction_details to the main thread
						tx.put((CLASSIFICATION_DONE, prediction_details))
						
						print("Class:", class_name)
						print("Confidence Score:", confidence_score)
						
					data = None
			
		except KeyboardInterrupt:
			rx.put((QUIT,))
		
		if RPI_PLATFORM:
			camera.close()
		else:
			camera.release()
		cv2.destroyAllWindows()
		print('[INFO] Camera service terminated!')
	
	def get_data(self):
		command = None
		data = None
		if not self.rx.empty():
			command, data = self.rx.get()
		return command, data
	
	def quit(self):
		self.tx.put((QUIT,))
	
	def stop(self):
		
		# self.grabber_process.terminate()
		self.grabber_process.join()
		
		del self.grabber_process
		
		del self.tx
		del self.rx
