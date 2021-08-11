# Run script: sudo ../venv/bin/python test.py

import cv2

from logic import *


cam_indxs = [0, 2, 4]

setup_cameras(cam_indxs)

camera_1 = Camera(cam_indxs[0], 20)
camera_2 = Camera(cam_indxs[1], 20)
camera_3 = Camera(cam_indxs[2], 30)

transporter_cntrl = TransporterControl('/dev/ttyUSB0')
text_dtctr = TextDetector(r'extra-files/token.json')
box_rcgnzr = BoxRecognizer('extra-files/boxes.json')

while True:
	if transporter_cntrl.get_state():
		frame_1 = camera_1.get_cropped_frame()
		frame_2 = camera_2.get_cropped_frame()
		frame_3 = camera_3.get_cropped_frame()

		text = text_dtctr.get_combined_text([frame_1, frame_2, frame_3])
		box_type = box_rcgnzr.get_type(text)

		print(box_type)

		cv2.imshow('Frame 1', frame_1)
		cv2.imshow('Frame 2', frame_2)
		cv2.imshow('Frame 3', frame_3)

	else:
		frame_1 = camera_1.get_frame()
		frame_2 = camera_2.get_frame()
		frame_3 = camera_3.get_frame()

		cv2.imshow('Live 1', frame_1)
		cv2.imshow('Live 2', frame_2)
		cv2.imshow('Live 3', frame_3)

	key = cv2.waitKey(1)

	if key == ord('q'):
		break

	elif key == ord('k'):
		camera_1.update_key_frame()
		camera_2.update_key_frame()
		camera_3.update_key_frame()

transporter_cntrl.stop()
