from twilio.rest import Client
#-------------- Twilio Credentials ------------
account_sid = ''
auth_token = ''
twilio_number = ''
rec_number =''	


def alert_sms(detection_type,cam_no,current_time):
    client = Client(account_sid,auth_token)

    message = client.messages.create(
        from_= twilio_number,
        to = rec_number,
        body = 'Camera '+ str(cam_no) +' has detected '+ detection_type + ' at ' + current_time)
    print(message.body)
    return True