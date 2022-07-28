from django.shortcuts import render
from django.http.response import HttpResponse, JsonResponse
from django.http import StreamingHttpResponse

#---------------------------- Libraries For Gun Detection --------------------------
import cv2
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from imutils.video import FPS
import winsound

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'MJPG', 'codec used in VideoWriter when saving video to file, MJPG or XVID')
flags.DEFINE_boolean('dis_cv2_window', True, 'disable cv2 window during the process') # this is good for the .ipynb


cam1 = cv2.VideoCapture(1)

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(cam1_frame(cam1), content_type='multipart/x-mixed-replace; boundary=frame')


#---------------------------- Gun & Fight Detection  --------------------------



#-------- Gun Detection -----------
def cam1_frame(cam1):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = 416
    video_path = 0
    gun_detect_count = 0
    detection_type = None
    try:
	    vid = cam1
    except:
    	vid = cv2.VideoCapture(int(video_path))
    saved_model_loaded = tf.saved_model.load('D:\FYP\Smart Surveillance System\smart_surveillance_system\surveillance_system\checkpoints\yolov4-tiny-416', tags=[tag_constants.SERVING])
    # saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        else:
            # print("------------------------")
            # print(return_value)
            # print("------------------------")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            if valid_detections.numpy() == 1:
                gun_detect_count = gun_detect_count + 1
                if gun_detect_count == 5:
                    detection_type = 'Gun'
                    generate_alarm(detection_type)
                    gun_detect_count = 0
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_id += 1
            ret, buffer = cv2.imencode('.jpg', result)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
            

#---------- Generate Alarm ------------
def generate_alarm(detection_type):
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration) 
    print(detection_type)  

