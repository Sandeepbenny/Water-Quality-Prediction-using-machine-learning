import re
from django.shortcuts import render
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from django.shortcuts import render

# Create your views here.

def LoginView(request):
    return render(request,'login.html')

def verifyloginview(request):
    if request.method == 'POST':
        username=request.POST["username"]
        password=request.POST["password"]
        if username =='admin' and password =='admin':
           return render(request,'home.html')
        else:
           return render(request,'login.html')