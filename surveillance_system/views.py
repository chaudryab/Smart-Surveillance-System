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
import datetime
from .helpers import *
from .send_sms import *
from django.conf import settings
from twilio.rest import Client
from django.contrib.auth.hashers import check_password, make_password
from django.contrib.auth import update_session_auth_hash
import random
from django.db.models import Q

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
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import load_model
from imutils.video import FPS
import winsound
import os
import tensorflow.lite as tflite
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
#--------- Model Load ---------
gun_model =  os.path.join(settings.BASE_DIR,'gun_detection/checkpoints/yolov4-tiny-416')
fight_model =  os.path.join(settings.BASE_DIR,'fight_detection/fightModel.tflite')
#--------- Global Variables ---------
global cam1
global cam2
global cam1_mode
global cam2_mode
global input1
global input2
#--------- Stream Input ---------
input1=int(2)
input2=int(3)
# input2='http://192.168.1.6:4747/video'
cam1=cv2.VideoCapture(input1)
cam2=cv2.VideoCapture(input2)
#--------- No. Camera Mode ---------
cam1_mode=int(1)
cam2_mode=int(0)
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
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
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
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
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
    gun_counts=Log.objects.filter(status=1,detection_type='Gun').count()
    fight_counts=Log.objects.filter(status=1,detection_type='Fight').count()
    detection = {'gun_counts':gun_counts,'fight_counts':fight_counts}
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    return render(request, 'index.html',detection)

#------------- Gun Detection --------------
@login_required
def gun_detect(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    cam_status = {'cam1':cam1_mode,'cam2':cam2_mode}
    return render(request, 'gun_detect.html',cam_status)

#------------- Fight Detection --------------
@login_required
def fight_detect(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    cam_status = {'cam1':cam1_mode,'cam2':cam2_mode}
    return render(request, 'fight_detect.html',cam_status)

#------------- Gun & Fight Detection --------------
@login_required
def gun_fight_detect(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    cam_status = {'cam1':cam1_mode,'cam2':cam2_mode}
    return render(request, 'gun&fight_detect.html',cam_status)

#------------- Camera 1 Video Feed For Gun Detection --------------
@login_required
def cam1_video_feed(request):
    return StreamingHttpResponse(cam1_gun_detect(cam1,input1), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Camera 2 Video Feed For Gun Detection --------------
@login_required
def cam2_video_feed(request):
    return StreamingHttpResponse(cam2_gun_detect(cam2,input2), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Camera 1 Video Feed For Fight Detection --------------
@login_required
def cam1_fight_video_feed(request):
    return StreamingHttpResponse(cam1_fight_detect(cam1,input1), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Camera 2 Video Feed For Fight Detection --------------
@login_required
def cam2_fight_video_feed(request):
    return StreamingHttpResponse(cam2_fight_detect(cam2,input2), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Camera 1 Video Feed For Gun & Fight Detection --------------
@login_required
def cam1_gun_fight_video_feed(request):
    return StreamingHttpResponse(cam1_gun_fight_detect(cam1,input1), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Camera 2 Video Feed For Gun & Fight Detection --------------
@login_required
def cam2_gun_fight_video_feed(request):
    return StreamingHttpResponse(cam2_gun_fight_detect(cam2,input2), content_type='multipart/x-mixed-replace; boundary=frame')

#------------- Log Alerts --------------
@login_required
def alert_logs(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    logs = Log.objects.filter(status=1).order_by('-id')
    logs={'logs':logs}
    return render(request,'alert_logs.html',logs)

#------------- Gun Detecion Logs --------------
@login_required
def gun_logs(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    logs = Log.objects.filter(status=1,detection_type='Gun').order_by('-id')
    logs={'logs':logs}
    return render(request,'gun_logs.html',logs)

#------------- Fight Detecion Logs --------------
@login_required
def fight_logs(request):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    logs = Log.objects.filter(status=1,detection_type='Fight').order_by('-id')
    logs={'logs':logs}
    return render(request,'fight_logs.html',logs)

#------------- Log Delete --------------
@login_required
def view_log(request,pk):
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
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

#------------- Gun Detections Per Month Graph In Admin Panel --------------
@login_required
def GunDetectionChart(request):
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    labels = []
    gun_detection_count = []
    thisYear = True
    current_date = datetime.date.today()
    current_month = current_date.month
    current_year = current_date.year
    i=0
    while i<12:
        month = (current_month-i)-1
        if month < 0:
            month = month+12
            thisYear = False
        if thisYear == True:
            date = month_names[month] +' '+ str(current_year)
            gun_log_count = Log.objects.filter(Q(created_at__year=current_year) & Q(created_at__month=month+1) & Q(detection_type='Gun')).count()
            gun_detection_count.append(str(gun_log_count))
        else:
            date = month_names[month] +' '+str(current_year-1)
            gun_log_count = Log.objects.filter(Q(created_at__year=current_year-1) & Q(created_at__month=month+1) & Q(detection_type='Gun')).count()
            gun_detection_count.append(str(gun_log_count))
        i = i+1
        labels.append(date)
    
    return JsonResponse({'months': labels, 'gun_detection_count': gun_detection_count}, safe=False)

#------------- Fight Detections Per Month Graph In Admin Panel --------------
@login_required
def FightDetectionChart(request):
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    labels = []
    fight_detection_count = []
    thisYear = True
    current_date = datetime.date.today()
    current_month = current_date.month
    current_year = current_date.year
    i=0
    while i<12:
        month = (current_month-i)-1
        if month < 0:
            month = month+12
            thisYear = False
        if thisYear == True:
            date = month_names[month] +' '+ str(current_year)
            fight_log_count = Log.objects.filter(Q(created_at__year=current_year) & Q(created_at__month=month+1) & Q(detection_type='Fight')).count()
            fight_detection_count.append(str(fight_log_count))
        else:
            date = month_names[month] +' '+str(current_year-1)
            fight_log_count = Log.objects.filter(Q(created_at__year=current_year-1) & Q(created_at__month=month+1) & Q(detection_type='Fight')).count()
            fight_detection_count.append(str(fight_log_count))
        i = i+1
        labels.append(date)
    
    return JsonResponse({'months': labels, 'fight_detection_count': fight_detection_count}, safe=False)

#---------------------------- Gun & Fight Detection  --------------------------

#-------- Camera 1 Gun Detection -----------
def cam1_gun_detect(cam1,input1):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    video_path = 0
    gun_detect_count = 0
    detection_type = None
    cam_no = 1
    cam1=cv2.VideoCapture(input1)
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
def cam2_gun_detect(cam2,input2):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    video_path = 0
    gun_detect_count = 0
    detection_type = None
    cam_no = 2
    cam2=cv2.VideoCapture(input2)
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
  
#-------- Camera 1 Fight Detection -----------
def cam1_fight_detect(cam1,input1):
    video_path = 0
    frame_count = 0
    fight_detect_count = 0
    detection_type = None
    cam_no = 1
    class_ind = {
    0: 'fight',
    1: 'nofight'
    }
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(fight_model)
    # allocate the tensors
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    cam1=cv2.VideoCapture(input1)
    try:
	    vid = cam1
    except:
    	vid = cv2.VideoCapture(int(video_path))
    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        else:
            imgF = cv2.resize(frame, (224, 224))
            normalized_frame = imgF / 255
            # Preprocess the image to required size and cast
            input_shape = input_details[0]['shape']
            input_tensor = np.array(np.expand_dims(normalized_frame, 0), dtype=np.float32)
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.squeeze(output_data)
            highest_pred_loc = np.argmax(pred)
            bird_name = class_ind[highest_pred_loc]
            if bird_name == 'fight':
                fight_detect_count = fight_detect_count + 1
            if frame_count == 30:
                frame_count = 0
                if fight_detect_count >= 25:
                    detection_type = 'Fight'
                    num = random.random()
                    cv2.imwrite(f"static/detection_images/frame_{num}.jpg", frame)
                    detection_log(detection_type,cam_no,num)
                    fight_detect_count = 0
                else:
                    fight_detect_count = 0
            frame_id += 1
            frame_count = frame_count+1
            ret, buffer = cv2.imencode('.jpg', frame)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
            
    
#-------- Camera 2 Fight Detection -----------
def cam2_fight_detect(cam2,input2):
    video_path = 0
    frame_count = 0
    fight_detect_count = 0
    detection_type = None
    cam_no = 2
    class_ind = {
    0: 'fight',
    1: 'nofight'
    }
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(fight_model)
    # allocate the tensors
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    cam2=cv2.VideoCapture(input2)
    try:
	    vid = cam2
    except:
    	vid = cv2.VideoCapture(int(video_path))
    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        else:
            imgF = cv2.resize(frame, (224, 224))
            normalized_frame = imgF / 255
            # Preprocess the image to required size and cast
            input_shape = input_details[0]['shape']
            input_tensor = np.array(np.expand_dims(normalized_frame, 0), dtype=np.float32)
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.squeeze(output_data)
            highest_pred_loc = np.argmax(pred)
            bird_name = class_ind[highest_pred_loc]
            if bird_name == 'fight':
                fight_detect_count = fight_detect_count + 1
            if frame_count == 30:
                frame_count = 0
                if fight_detect_count >= 25:
                    detection_type = 'Fight'
                    num = random.random()
                    cv2.imwrite(f"static/detection_images/frame_{num}.jpg", frame)
                    detection_log(detection_type,cam_no,num)
                    fight_detect_count = 0
                else:
                    fight_detect_count = 0
            frame_id += 1
            frame_count = frame_count+1
            ret, buffer = cv2.imencode('.jpg', frame)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
    

#-------- Camera 1 Gun & Fight Detection -----------
def cam1_gun_fight_detect(cam1,input1):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    video_path = 0
    frame_count = 0
    fight_detect_count = 0
    gun_detect_count = 0
    detection_type = None
    cam_no = 1
     # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path= fight_model)
    # allocate the tensors
    interpreter.allocate_tensors()
    cam1=cv2.VideoCapture(input1)
    class_ind = {
    0: 'fight',
    1: 'nofight'
    }
    try:
	    vid = cam1
    except:
    	vid = cv2.VideoCapture(int(video_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    saved_model_loaded = tf.saved_model.load(gun_model,tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        else:
            imgF = cv2.resize(frame, (224, 224))
            normalized_frame = imgF / 255
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
            # Preprocess the image to required size and cast
            input_shape = input_details[0]['shape']
            input_tensor = np.array(np.expand_dims(normalized_frame, 0), dtype=np.float32)
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.squeeze(output_data)
            highest_pred_loc = np.argmax(pred)
            bird_name = class_ind[highest_pred_loc]
            if bird_name == 'fight':
                fight_detect_count = fight_detect_count + 1
            if frame_count == 30:
                frame_count = 0
                if fight_detect_count >= 25:
                    detection_type = 'Fight'
                    num = random.random()
                    cv2.imwrite(f"static/detection_images/frame_{num}.jpg", frame)
                    detection_log(detection_type,cam_no,num)
                    fight_detect_count = 0
                else:
                    fight_detect_count = 0
            frame_id += 1
            frame_count = frame_count+1
            frame_id += 1
            ret, buffer = cv2.imencode('.jpg', result)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
    

#-------- Camera 2 Gun & Fight Detection -----------
def cam2_gun_fight_detect(cam1,input2):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    video_path = 0
    frame_count = 0
    fight_detect_count = 0
    gun_detect_count = 0
    detection_type = None
    cam_no = 2
     # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path= fight_model)
    # allocate the tensors
    interpreter.allocate_tensors()
    cam1=cv2.VideoCapture(input2)
    class_ind = {
    0: 'fight',
    1: 'nofight'
    }
    try:
	    vid = cam1
    except:
    	vid = cv2.VideoCapture(int(video_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    saved_model_loaded = tf.saved_model.load(gun_model,tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        else:
            imgF = cv2.resize(frame, (224, 224))
            normalized_frame = imgF / 255
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
            # Preprocess the image to required size and cast
            input_shape = input_details[0]['shape']
            input_tensor = np.array(np.expand_dims(normalized_frame, 0), dtype=np.float32)
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.squeeze(output_data)
            highest_pred_loc = np.argmax(pred)
            bird_name = class_ind[highest_pred_loc]
            if bird_name == 'fight':
                fight_detect_count = fight_detect_count + 1
            if frame_count == 30:
                frame_count = 0
                if fight_detect_count >= 25:
                    detection_type = 'Fight'
                    num = random.random()
                    cv2.imwrite(f"static/detection_images/frame_{num}.jpg", frame)
                    detection_log(detection_type,cam_no,num)
                    fight_detect_count = 0
                else:
                    fight_detect_count = 0
            frame_count = frame_count+1
            ret, buffer = cv2.imencode('.jpg', result)
            fframe = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fframe + b'\r\n')
    
 

#---------- Detection Log Save In DB ------------
def detection_log(detection_type,cam_no,num):
    from datetime import datetime, date
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = date.today() 
    log = Log(image=num,cam_no=cam_no,detection_type=detection_type,time=current_time,date=current_date)
    log.save()
    generate_alarm()
    alert_sms(detection_type,cam_no)
    # alert_mail(detection_type,cam_no,current_time,current_date,num)
    
    
#---------- Generate Alarm ------------
def generate_alarm():
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration) 

