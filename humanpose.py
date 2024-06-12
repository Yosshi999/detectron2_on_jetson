import os
import time

import numpy as np
import cv2
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

model_name = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x"
cfg_name = model_name + ".yaml"

def draw_humanpose(frame: np.ndarray, keypoints: np.ndarray, metadata) -> np.ndarray:
	"""Draw humanpose on the frame in-place.

	Parameters:
		frame: uint8 array (H, W, 3)
			RGB image.
		keypoints: float array (17, 3)
			Keypoints of one human given by detectron2. [x, y, visibility].
		metadata: dict
			Metadata of trained model.
	Returns:
		frame: uint8 array (H, W, 3)
			Annotated image.
	"""
	threshold = 0.05

	visible = {}
	keypoint_names = metadata.get("keypoint_names")
	connection_rules = metadata.get("keypoint_connection_rules")
	assert keypoint_names
	assert connection_rules

	for idx, keypoint in enumerate(keypoints):
		x, y, prob = keypoint
		if prob > threshold:
			x = int(x)
			y = int(y)
			cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
			if keypoint_names:
				keypoint_name = keypoint_names[idx]
				visible[keypoint_name] = (x, y)

	for kp0, kp1, color in connection_rules:
		if kp0 in visible and kp1 in visible:
			x0, y0 = visible[kp0]
			x1, y1 = visible[kp1]
			cv2.line(frame, (x0, y0), (x1, y1), color, 3)
	# draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
	# Note that this strategy is specific to person keypoints.
	# For other keypoints, it should just do nothing
	try:
		ls_x, ls_y = visible["left_shoulder"]
		rs_x, rs_y = visible["right_shoulder"]
		mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) // 2, (ls_y + rs_y) // 2
	except KeyError:
		pass
	else:
		# draw line from nose to mid-shoulder
		nose_x, nose_y = visible.get("nose", (None, None))
		if nose_x is not None:
			cv2.line(frame, (nose_x, nose_y), (mid_shoulder_x, mid_shoulder_y), (0, 0, 255), 3)
		try:
			# draw line from mid-shoulder to mid-hip
			lh_x, lh_y = visible["left_hip"]
			rh_x, rh_y = visible["right_hip"]
		except KeyError:
			pass
		else:
			mid_hip_x, mid_hip_y = (lh_x + rh_x) // 2, (lh_y + rh_y) // 2
			cv2.line(frame, (mid_hip_x, mid_hip_y), (mid_shoulder_x, mid_shoulder_y), (0, 0, 255), 3)
	return frame


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
cfg.MODEL.WEIGHTS = os.path.join("./weights", model_name + ".pkl")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

cap = cv2.VideoCapture(0)
assert cap.isOpened()
ret, frame = cap.read()
assert ret
print("original frame size:", frame.shape[1], "x", frame.shape[0])

measure_start = time.perf_counter()
processed_frames = 0
fps = 0
while True:
	ret, frame = cap.read()
	if not ret:
		print("unable to fetch camera frame")
		break
	
	outputs = predictor(frame)
	view = frame.copy()
	for keypoints in outputs["instances"].pred_keypoints:
		draw_humanpose(view, keypoints, metadata)

	view = view[:, ::-1, :]  # horizontal flip
	view = np.ascontiguousarray(view)
	cv2.putText(view, "push ESC to exit", (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3, cv2.LINE_AA)
	cv2.putText(view, f"FPS: {fps:.2f}", (0, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3, cv2.LINE_AA)

	processed_frames += 1
	measure_time = time.perf_counter() - measure_start
	if processed_frames >= 10 or measure_time >= 1:
		fps = measure_time / processed_frames
		measure_start = time.perf_counter()
		processed_frames = 0

	cv2.imshow("camera", view)
	if cv2.waitKey(10) == 27:
		break

cap.release()
cv2.destroyAllWindows()
