from django.db import models

# Create your models here.

#------------- Detection Log Table --------------
class Log(models.Model):
    id = models.AutoField(primary_key=True)
    cam_no = models.IntegerField(default=None)
    detection_type = models.CharField(max_length=255)
    time = models.TimeField(null=True)
    date = models.DateField(null=True)
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=True)