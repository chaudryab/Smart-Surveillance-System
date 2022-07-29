from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags

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

