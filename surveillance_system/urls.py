from django.urls import path
from . import views

urlpatterns = [
    path('',views.login,name='login'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('change_password',views.change_password,name='change_password'),
    path('forget_pwd',views.forget_pwd,name='forget_pwd'),
    path('reset_pwd/<str:token>',views.reset_pwd,name='reset_pwd'),
    path('error_404',views.error_404,name='error_404'),
    path('success',views.success,name='success'),
    path('index',views.index,name='index'),
    path('gun_detect',views.gun_detect,name='gun_detect'),
    path('fight_detect',views.fight_detect,name='fight_detect'),
    path('gun_fight_detect',views.gun_fight_detect,name='gun_fight_detect'),
    path('cam1_video_feed',views.cam1_video_feed,name='cam1_video_feed'),
    path('cam2_video_feed',views.cam2_video_feed,name='cam2_video_feed'),
    path('cam1_fight_video_feed',views.cam1_fight_video_feed,name='cam1_fight_video_feed'),
    path('cam2_fight_video_feed',views.cam2_fight_video_feed,name='cam2_fight_video_feed'),
    path('cam1_gun_fight_video_feed',views.cam1_gun_fight_video_feed,name='cam1_gun_fight_video_feed'),
    path('cam2_gun_fight_video_feed',views.cam2_gun_fight_video_feed,name='cam2_gun_fight_video_feed'),
    path('alert_logs',views.alert_logs,name='alert_logs'),
    path('gun_logs',views.gun_logs,name='gun_logs'),
    path('fight_logs',views.fight_logs,name='fight_logs'),
    path('view_log/<pk>',views.view_log,name='view_log'),
    path('share_log/<pk>',views.share_log,name='share_log'),
    path('del_log/<pk>',views.del_log,name='del_log'),
    path('monthly_gun_detection_chart', views.GunDetectionChart, name="monthly_gun_detection_chart"),
    path('monthly_fight_detection_chart', views.FightDetectionChart, name="monthly_fight_detection_chart"),
]