from django.db import models

# -*- coding: utf-8 -*-
 
from django import forms
# Create your models here.
from django.db import models
#define common max_length for all
lenn=100
#
 
#User model
from django.contrib.auth.models import User
# Create your models here.

from django.urls import reverse 


class comment_model(models.Model):
	comment = models.CharField("comment input",
		max_length=lenn)

class Post(models.Model) :
	text = models.TextField()
	author  = models.ForeignKey(User, on_delete = models.CASCADE)

	def __str__(self) :
		return self.text

	def get_absolute_url(self) :
		return reverse('home-view')