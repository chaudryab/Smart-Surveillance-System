from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import auth
from django.contrib.auth.models import User
import uuid
from django.http.response import HttpResponse, JsonResponse
from django.http import HttpResponseRedirect
from django.http import StreamingHttpResponse
from .models import *
from datetime import datetime, date
from .helpers import *
from django.contrib.auth.hashers import check_password, make_password
from django.contrib.auth import update_session_auth_hash
import random

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
from keras.models import load_model
from imutils.video import FPS
import winsound
import os

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

gun_model =  os.path.join(settings.BASE_DIR,'gun_detection/checkpoints/yolov4-tiny-416')

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(0)


#---------------------------- Admin Portal  --------------------------

#------------- Login --------------
@csrf_exempt
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username, password=password)
        if user:
            if user.is_superuser:
                auth.login(request, user)
                return redirect('index')
        else:
            messages.error(request, 'Invalid Crendentials')
            return redirect('login')
    else:
        return render(request, 'login.html')

#------------- Forget Password --------------
@csrf_exempt
def forget_pwd(request):
    if request.method == 'POST':
        email = request.POST['email']
        superuser = User.objects.get(is_superuser=True)
        if superuser.email == email:
            token = str(uuid.uuid4())
            admin_token = Admin_token.objects.get(id=1)
            admin_token.token = token
            admin_token.save()
            send_admin_forget_password_mail(email,token)
            messages.success(request, 'Email Sent!!')
            return redirect('forget_pwd')
        else:
            messages.error(request, 'Email Not Exist!!')
            return redirect('forget_pwd')
    return render(request,'forget_pwd.html')

#------------- Reset Password --------------
@csrf_exempt
def reset_pwd(request,token):
    if request.method == 'GET':
        if Admin_token.objects.filter(token=token).exists():
            return render(request, 'reset_pwd.html')
        else:
            return redirect('error_404')
    else:
        if request.method == 'POST':
            new_password = request.POST['new_password']
            confirm_password = request.POST['confirm_password']
            superusers = User.objects.get(is_superuser=True)
            if int(len(new_password)) < 6:
                messages.error(request, 'Password Must Contains Six Characters!!')
                return HttpResponseRedirect(request.META.get('HTTP_REFERER','/'))
            elif new_password == confirm_password:
                super_pwd = make_password(new_password, None, 'md5')
                superusers = User.objects.get(is_superuser=True)
                superusers.password = super_pwd
                admin_token = Admin_token.objects.filter(id=1).first()
                admin_token.token = None
                superusers.save()
                admin_token.save()
                messages.success(request, 'Password Changed!!')
                return redirect('success')
            else:
                messages.error(request, 'Password Did Not Match!!')
                return HttpResponseRedirect(request.META.get('HTTP_REFERER','/'))
        else:
            return redirect('error_404')

#------------- Admin Change Password --------------
@login_required
@csrf_exempt
def change_password(request):
    superusers = User.objects.get(is_superuser=True)
    if request.method == 'POST':
        old_password = request.POST['old_password']
        new_password = request.POST['new_password']
        confirm_password = request.POST['confirm_password']
        superusers = User.objects.get(is_superuser=True)
        if check_password(old_password,superusers.password):
            if int(len(new_password)) < 6:
                messages.error(request, 'Password Must Contains Six Characters!!')
                return redirect('change_password')
            elif new_password == confirm_password:
                super_pwd = make_password(new_password, None, 'md5')
                superusers.password = super_pwd
                superusers.save()
                messages.success(request, 'Password successfully changed.')
                update_session_auth_hash(request, superusers)
                return redirect('change_password')
            else:
                messages.error(request, 'Password Did Not Match!!')
                return redirect('change_password')
        else:
            messages.error(request, 'Invalid Old Password!!')
            return redirect('change_password')
    return render(request,'change_password.html')

#------------- Admin Logout --------------
@login_required
def logout(request):
    auth.logout(request)
    return redirect('login')

#------------- Error-404 Page --------------
def error_404(request):
    return render(request,'error_404.html')

#------------- Success Page--------------
def success(request):
    return render(request,'success.html')

#------------- Dashboard --------------
@login_required
def index(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    return render(request, 'index.html')

#------------- Gun Detection --------------
@login_required
def gun_detect(request):
    cam_status = {'cam1':1,'cam2':0}
    return render(request, 'gun_detect.html',cam_status)

#------------- Camera 1 Video Feed For Gun Detection --------------
@login_required
def cam1_video_feed(request):
    return StreamingHttpResponse(cam1_gun_detect(cam1), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Camera 2 Video Feed For Gun Detection --------------
@login_required
def cam2_video_feed(request):
    return StreamingHttpResponse(cam2_gun_detect(cam2), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Log Alerts --------------
@login_required
def alert_logs(request):
    logs = Log.objects.filter(status=1).order_by('-id')
    logs={'logs':logs}
    return render(request,'alert_logs.html',logs)

#------------- Log Delete --------------
@login_required
def view_log(request,pk):
    logs=Log.objects.filter(id=pk).first()
    log={'log':logs}
    return render(request,'view_log.html',log)

#------------- Log Delete --------------
@login_required
def del_log(request,pk):
    log=Log.objects.filter(id=pk).first()
    log.status=0
    log.save()
    messages.success(request,"Log Deleted Successfully !!")
    return redirect('alert_logs')


#---------------------------- Gun & Fight Detection  --------------------------

#-------- Camera 1 Gun Detection -----------
def cam1_gun_detect(cam1):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    video_path = 0
    gun_detect_count = 0
    detection_type = None
    cam_no = 1
    try:
	    vid = cam1
    except:
    	vid = cv2.VideoCapture(int(video_path))
    saved_model_loaded = tf.saved_model.load(gun_model,tags=[tag_constants.SERVING])
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
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
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
                    num = random.random()
                    frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"static/detection_images/frame_{num}.jpg", frame_in_rgb)
                    detection_log(detection_type,cam_no,num)
                    gun_detect_count = 0
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_id += 1
            ret, buffer = cv2.imencode('.jpg', result)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
    
            
#-------- Camera 2 Gun Detection -----------
def cam2_gun_detect(cam2):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    video_path = 0
    gun_detect_count = 0
    detection_type = None
    cam_no = 2
    try:
	    vid = cam2
    except:
    	vid = cv2.VideoCapture(int(video_path))
    saved_model_loaded = tf.saved_model.load(gun_model,tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
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
                    num = random.random()
                    frame_in_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"static/detection_images/frame_{num}.jpg", frame_in_rgb)
                    detection_log(detection_type,cam_no,num)
                    gun_detect_count = 0
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_id += 1
            ret, buffer = cv2.imencode('.jpg', result)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
            

#---------- Detection Log Save In DB ------------
def detection_log(detection_type,cam_no,num):
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = date.today() 
    log = Log(image=num,cam_no=cam_no,detection_type=detection_type,time=current_time,date=current_date)
    log.save()
    generate_alarm()
    
#---------- Generate Alarm ------------
def generate_alarm():
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration) 

