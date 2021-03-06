# Файл logic.py содержит основную логику программы

import threading, time, os, json

import cv2
import serial

from skimage.metrics import structural_similarity
from google.cloud import vision
from fuzzywuzzy import fuzz


# Вспомогательный класс для отображения исключений
class ExceptionPrinter:

	def __init__(self):
		self.prev_exception = None

	def print(self, exception):
		exception = str(exception)

		if exception != self.prev_exception:
			print(exception)
			self.prev_exception = exception


ex_printer = ExceptionPrinter()


# Класс для получения кадров с веб-камеры
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
		return self.frame_cropper.get_cropped(self.key_frame, frame)

	def update_key_frame(self):
		self.key_frame = cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2GRAY)

	def stop(self):
		self.capture.release()


# Класс для детектирования коробочки в кадре, и обрезки этого кадра
# Сейчас реализованы два лучших метода детектирования коробочки:
#   - с помощью библиотеки skimage;
#   - с помощью библиотеки opencv.
# Были попытки реализовать это с помощью других алгоритмов, но точность была ниже, нежели при использовании этих алгоритмов
# Лучшая точность сейчас получается с библиотекой skimage
class FrameCropper:

	def __init__(self):
		pass

	def get_cropped(self, key_frame, frame):
		return self.skimage_processing(key_frame, frame) # сейчас используются алгоритмы библиотеки skimage

	def skimage_processing(self, key_frame, frame):
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		difference = structural_similarity(key_frame, gray_frame, full=True)[1]
		difference = (difference * 255).astype('uint8')
		threshold = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		x, y, w, h = self.get_max_contour(contours)

		return frame[y:y+h, x:x+w]

	def opencv_processing(self, key_frame, frame):
		new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		new_frame = cv2.GaussianBlur(new_frame, (21, 21), 0)

		delta = cv2.absdiff(key_frame, new_frame)
		threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
		threshold = cv2.dilate(threshold, None, iterations=2)
		contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		x, y, w, h = self.get_max_contour(contours)

		return frame[y:y+h, x:x+w]

	def get_max_contour(self, contours):
		max_contour = [0, 0, 0, 0]

		for c in contours:
			x, y, w, h = cv2.boundingRect(c)

			current_area = w * h
			max_area = max_contour[2] * max_contour[3]

			if current_area > max_area:
				max_contour = [x, y, w, h]

		return max_contour


# Класс для управления Транспортером при помощи сериал порта
class TransporterControl:

	def __init__(self, port_name):
		self.port_name = port_name
		self.open_port()

		self.is_thread = True
		self.thread = threading.Thread(target=self.update)

		self.is_ready = False

		self.responses = [
			b'info: transporter C box ready to take\n'
		]

		self.thread.start()

	def open_port(self):
		try:
			self.port = serial.Serial(port=self.port_name)
		except Exception as e:
			ex_printer.print(e)
			self.port = None

	def update(self):
		while self.is_thread:
			try:
				current_msg = self.port.readline()

				if current_msg == self.responses[0]:
					time.sleep(1)
					self.is_ready = True
			except Exception as e:
				ex_printer.print(e)
				self.is_ready = False

	def get_state(self):
		if self.is_ready:
			self.is_ready = False
			return True
		else:
			return False

	def stop(self):
		self.is_thread = False
		self.thread.join()


# Класс для детектирования текста на коробочке с помощью сервисов Google
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


# Класс для распознавания типа лекарства
# В файле extra-files/boxes.json записаны ключевые слова для лекарств
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
						#print(f'{config_word} - {word} - {current_rating}')

			if rating > max_rating[1]:
				max_rating = [key, rating]

		return max_rating[0]
