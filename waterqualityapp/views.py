import re
from django.shortcuts import render
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,consensus_score
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

def prediction(request):
     return render(request,'prediction.html')
 
def processdataview(request):
    if request.method == 'POST':
        print('reached')    
        PH=request.POST['PH']
        Hardness=request.POST['Hardness']
        Solids=request.POST['Solids']
        Chloramines=request.POST['Chloramines']
        Sulfate=request.POST['Sulfate']
        Conductivity=request.POST['Conductivity']
        Organic_carbon=request.POST['Organic_carbon']
        Trihalomethanes=request.POST['Trihalomethanes']
        Turbidity=request.POST['Turbidity']
        mydata=[[PH,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]]
        dataset = pd.read_csv('water_potability.csv')
        dataset.ph.fillna(value=dataset.ph.mean(),inplace=True)
        dataset.Hardness.fillna(value=dataset.Hardness.mean(),inplace=True)
        dataset.Solids.fillna(value=dataset.Solids.mean(),inplace=True)
        dataset.Chloramines.fillna(value=dataset.Chloramines.mean(),inplace=True)
        dataset.Sulfate.fillna(value=dataset.Sulfate.mean(),inplace=True)
        dataset.Conductivity.fillna(value=dataset.Conductivity.mean(),inplace=True)
        dataset.Organic_carbon.fillna(value=dataset.Organic_carbon.mean(),inplace=True)
        dataset.Trihalomethanes.fillna(value=dataset.Trihalomethanes.mean(),inplace=True)
        dataset.Turbidity.fillna(value=dataset.Turbidity.mean(),inplace=True)
        model=DecisionTreeClassifier(criterion='entropy',random_state=0)
        # Y_predicted_dt=model.predict(X_test)
        # confusion_matrix(Y_test,Y_predicted_dt)
        # accuracy_score(Y_test,Y_predicted_dt)
        # model=DecisionTreeClassifier(criterion='entropy',random_state=0)
        X_array=np.array(dataset.iloc[:,:-1])
        Y=np.array(dataset.iloc[:,-1])
        Y=Y.reshape(-1,1)
        X_train, X_test, Y_train, Y_test = train_test_split (X_array,Y,test_size=0.25,random_state=42)
        model.fit(X_train,Y_train)
        result=model.predict(mydata)
        print(result)
        #return render(request,'home.html')
        if(result[0] < 0.5):
            prediction= 'IMPURE'
            print("water NEGATIVE")
        else:
            print("water POSITIVE")
            prediction= 'PURE'
    return render(request,'Result.html',{'result':prediction})

def Result(request):
     return render(request,'Result.html')