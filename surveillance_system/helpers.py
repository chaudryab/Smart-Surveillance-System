from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.core.mail import EmailMultiAlternatives
import os

#------------- Forget Password --------------
def send_admin_forget_password_mail(email,token):
    subject = 'your change password link'
    link = 'http://127.0.0.1:8000/surveillance_system/reset_pwd/'+token
    html_message = render_to_string('mail_template.html', {'link': link})
    plain_message = strip_tags(html_message)
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]
    send_mail(subject, plain_message, email_from, recipient_list, html_message=html_message)
    return True


#------------- Detection Alert Email --------------
def alert_mail(detection_type,cam_no,current_time,current_date,num):
    subject, from_email, to = 'Detection Alert - Smart Surveillance System', 'abchaudry9@gmail.com', 'abchaudry9@gmail.com'
    html_content = render_to_string('alert_mail.html', {'detection_type': detection_type,'cam_no':cam_no,'current_time':current_time,'current_date':current_date}).strip()
    msg = EmailMultiAlternatives(subject,html_content, from_email, [to])
    msg.content_subtype = 'html'
    detection_image =  os.path.join(settings.BASE_DIR,f"static/detection_images/frame_{num}.jpg")
    msg.attach_file(detection_image)
    msg.send()
    return True