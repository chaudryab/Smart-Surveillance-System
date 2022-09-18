from twilio.rest import Client

account_sid = 'AC016f26a047544f764cdd928c43e94408'
auth_token = '3596291347aa6d0b748e27c1afd33096'
twilio_number = '+15739953870'
rec_number ='+923084080900'

def alert_sms(detection_type,cam_no):
    client = Client(account_sid,auth_token)

    message = client.messages.create(
        from_= twilio_number,
        to = rec_number,
        body = 'Camera '+ str(cam_no) +' has detected '+ detection_type)
    print(message.body)
    return True
