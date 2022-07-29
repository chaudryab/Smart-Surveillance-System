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
    path('video_feed',views.video_feed,name='video_feed'),
]