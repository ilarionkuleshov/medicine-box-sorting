import threading, time, os, json

import cv2
import serial

from skimage.metrics import structural_similarity
from google.cloud import vision
from fuzzywuzzy import fuzz


class Camera:

	def __init__(self, index, focus_value):
		os.system(f'v4l2-ctl -d /dev/video{index} --set-ctrl=focus_auto=0')
		os.system(f'v4l2-ctl -d /dev/video{index} --set-ctrl=exposure_auto=1')
		time.sleep(2)

		self.capture = cv2.VideoCapture(index)
		self.capture.set(28, focus_value)

		self.frame_cropper = FrameCropper()

		self.update_key_frame()

	def get_frame(self):
		return self.capture.read()[1]

	def get_cropped_frame(self):
		frame = self.get_frame()
		return self.frame_cropper.get_difference(self.key_frame, frame)

	def update_key_frame(self):
		self.key_frame = cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2GRAY)


class FrameCropper:

	def __init__(self):
		pass

	def get_difference(self, key_frame, frame):
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		difference = structural_similarity(key_frame, gray_frame, full=True)[1]
		difference = (difference * 255).astype('uint8')
		threshold = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
		x, y, w, h = self.contour_size_sorting(contours)

		return frame[y:y+h, x:x+w]

	def contour_size_sorting(self, contours):
		max_contour = [0, 0, 0, 0]

		for c in contours:
			x, y, w, h = cv2.boundingRect(c)

			current_area = w * h
			max_area = max_contour[2] * max_contour[3]

			if current_area > max_area:
				max_contour = [x, y, w, h]

		return max_contour


class TransporterControl:

	def __init__(self, port_name):
		self.port = serial.Serial(port=port_name)

		self.is_thread = True
		self.thread = threading.Thread(target=self.update)

		self.is_ready = False

		self.responses = [
			b'info: transporter C box ready to take\n'
		]

		self.thread.start()

	def update(self):
		while self.is_thread:
			current_msg = self.port.readline()

			if current_msg == self.responses[0]:
				time.sleep(1)
				self.is_ready = True

	def get_state(self):
		if self.is_ready:
			self.is_ready = False
			return True
		else:
			return False

	def stop(self):
		self.is_thread = False
		self.thread.join()


class TextDetector:

	def __init__(self, token):
		os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = token

		self.client = vision.ImageAnnotatorClient()

	def get_text(self, frame):
		content = cv2.imencode('.jpg', frame)[1].tobytes()
		image = vision.Image(content=content)

		response = self.client.text_detection(image=image)

		result_text = []
		for text in response.text_annotations:
			result_text.append(text.description)

		return result_text

	def get_combined_text(self, frames):
		combined_text = []

		for frame in frames:
			combined_text += self.get_text(frame)

		return combined_text


class BoxRecognizer:

	def __init__(self, config_file):
		with open(config_file) as file:
			self.config_data = json.load(file)

	def get_type(self, text):
		max_rating = ['', 0]

		for key in self.config_data:
			rating = 0

			for config_word in self.config_data[key]:
				for word in text:
					current_rating = fuzz.token_sort_ratio(config_word, word)

					if current_rating >= 70:
						rating += current_rating
						print(f'{config_word} - {word} - {current_rating}')

			if rating > max_rating[1]:
				max_rating = [key, rating]

		return max_rating[0]
