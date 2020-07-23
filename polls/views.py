from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from sklearn.externals import joblib 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from itertools import chain 
from .models import Post
from django.views.generic import CreateView, ListView
from django.urls import reverse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance




def index2(request):
    return HttpResponse("Hello, world. You're at the polls index.")


#homepage 
def index3(request):
    return render(request, 'polls/index.html')

#form

from django.http import HttpResponseRedirect
from django.shortcuts import render

from django.shortcuts import render
from .forms import ContactForm

a1 = 'heyy how are you'
a1 = a1.split()
a2 = [1,2,3,4]
dict_ = dict(zip(a1,a2))




def index(request):
    vect = TfidfVectorizer(max_features=40000,stop_words='english')
    target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    data = pd.read_csv('train.csv')
    test_data = pd.read_csv('D:/T.Y.BTECH/BML/Project/test.csv')
    X = data.comment_text
    test_X = test_data.comment_text
    xt = vect.fit_transform(X) 
    yt = vect.transform(test_X)
    y_trans = data.iloc[:,2:8]
    X_train,X_test,y_train,y_test = train_test_split(xt,y_trans,test_size=0.3)
    input_comment=''
    output_class=None
    toxic  = None    
    severe_toxic = None
    obscene = None
    threat = None
    insult = None
    identity_hate = None
    
    '''
    if request.method == 'GET' :
        if request.GET['dropdown'] == 'KNN' :
            load_model = joblib.load('knn.pkl')
        if request.GET['dropdown'] == 'SVM' :
            load_model = joblib.load('lr.pkl')

'''
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            input_comment = form.cleaned_data['comment']
            algo_sel=form.cleaned_data['algo_field']
            print (algo_sel, input_comment)

        	#output_class = dict_[input_comment]
        	#output_class = [ 'violence', 'obscene', 'insult']
        	#print( input_comment )  
        	#print( output_class )

        input_comment1 = str(input_comment)
        input_comment1 = [input_comment1]
        input_comment1 = vect.transform(input_comment1)

        if(algo_sel == "logistic regression") :
            #load_model = joblib.load('D:/T.Y.BTECH/BML/Project/lr.pkl')
            from skmultilearn.problem_transform import ClassifierChain

            classifier = ClassifierChain(
                LogisticRegression(),
                require_dense = [False, True]
            )
            classifier.fit(X_train, y_train)
            output_class = classifier.predict_proba(input_comment1).toarray()

        elif(algo_sel == "KNN") :
            #load_model = joblib.load('knn.pkl')
            classifier = BinaryRelevance(
                LogisticRegression(),
                require_dense = [False, True]
            )
            classifier.fit(X_train, y_train)
            output_class = classifier.predict_proba(input_comment1).toarray()
        else :
            load_model = joblib.load('br_builtin.pkl') # SVM Classifier
            output_class = load_model.predict_proba(input_comment1).toarray()


        #output_class = load_model.predict_proba(input_comment1).toarray()
        print(output_class)
       # output_class = output_class.tolist()
        output_class = list(chain.from_iterable(output_class)) 
        toxic = output_class[0]
        severe_toxic = output_class[1]
        obscene = output_class[2]
        threat = output_class[3]
        insult = output_class[4]
        identity_hate = output_class[5]
        print (output_class)



     
           #return HttpResponseRedirect('/thanks/')
    else:
        form = ContactForm()

    context = dict()
    context['form'] = form
    context['input_comment'] = input_comment
    context['output_class1'] = toxic
    context['output_class2'] = severe_toxic
    context['output_class3'] = obscene
    context['output_class4'] = threat
    context['output_class5'] = insult
    context['output_class6'] = identity_hate
    return render(request, 'polls/index.html', context)



def post(request) :
    context1 = {
       posts1 : Post.objects.all() 
    }
    return render(request, 'polls/post.html', context1)


class PostListView(ListView) :
    model = Post
    template_name = 'polls/post.html'
    context_object_name = 'posts1'

class PostCreateView(CreateView) :
    model = Post
    fields = ['text']

    def form_valid(self, form) :
        form.instance.author = self.request.user
        return super().form_valid(form)

    def get_success_url(self) :
        return reverse('check')


# Fetching record from database

def check(request) :
    vect = TfidfVectorizer(max_features=40000,stop_words='english')
    target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    data = pd.read_csv('train.csv')
    test_data = pd.read_csv('D:/T.Y.BTECH/BML/Project/test.csv')
    X = data.comment_text
    test_X = test_data.comment_text
    xt = vect.fit_transform(X) 
    yt = vect.transform(test_X)
    y_trans = data.iloc[:,2:8]
    X_train,X_test,y_train,y_test = train_test_split(xt,y_trans,test_size=0.3) 
    input_comment=''
    output_class=None
    toxic  = None    
    severe_toxic = None
    obscene = None
    threat = None
    insult = None
    identity_hate = None
    posts = Post.objects.all()
    for post in posts :
        cmnt = post
    input_comment1 = str(cmnt)
    input_comment1 = [input_comment1]
    input_comment1 = vect.transform(input_comment1)
    from skmultilearn.problem_transform import ClassifierChain

    classifier = ClassifierChain(
    LogisticRegression(),
    require_dense = [False, True]
    )
    classifier.fit(X_train, y_train)
    output_class = classifier.predict_proba(input_comment1).toarray()

    #load_model = joblib.load('knn.pkl')
    #load_model = joblib.load('lr.pkl')
    #output_class = load_model.predict_proba(input_comment1).toarray()
    # output_class = output_class.tolist()
    output_class = list(chain.from_iterable(output_class)) 
    toxic = output_class[0]
    severe_toxic = output_class[1]
    obscene = output_class[2]
    threat = output_class[3]
    insult = output_class[4]
    identity_hate = output_class[5]
    print (output_class)

    context = dict()
    context['input_comment'] = input_comment
    context['output_class1'] = toxic
    context['output_class2'] = severe_toxic
    context['output_class3'] = obscene
    context['output_class4'] = threat
    context['output_class5'] = insult
    context['output_class6'] = identity_hate
    return render(request, 'polls/comment_details.html', context)

       
   


