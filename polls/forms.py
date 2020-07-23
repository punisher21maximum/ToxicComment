from django import forms
from django.forms import ModelForm
from django.db import models
from polls.models import Post

class postForm(ModelForm) :
    class meta :
        model = Post
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in iter(self.fields):
            self.fields[field].widget.attrs.update({
            'class': 'form-control border border-danger'
    })

class ContactForm(forms.Form):
    comment = forms.CharField(max_length=80)
    algo_CHOICES=[('KNN','KNN'),('logistic regression','logistic regression'),
    ('SVM','SVM'),]
    algo_field=forms.ChoiceField(choices=algo_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control form-control-lg text-info border border-primary'}))

    '''
    email = forms.EmailField(max_length=254)
    message = forms.CharField(
        max_length=2000,
        widget=forms.Textarea(),
        help_text='Write here your message!'
    )
    source = forms.CharField(       # A hidden input for internal use
        max_length=50,              # tell from which page the user sent the message
        widget=forms.HiddenInput()
    )
	'''
    def clean(self):
        cleaned_data = super(ContactForm, self).clean()
        comment = cleaned_data.get('comment')
        email = cleaned_data.get('email')
        message = cleaned_data.get('message')
        if not comment and not email and not message:
            raise forms.ValidationError('You have to write something!')