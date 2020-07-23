from django.urls import path

from . import views
from .views import PostCreateView,PostListView

urlpatterns = [
    path('', views.index, name='index'),
   	path('details/', views.check, name='check'),
    path('home/', PostCreateView.as_view(), name = 'home-view'),
    path('comments/', PostListView.as_view(), name = 'comments-view'),

]