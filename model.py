import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('breast-cancer.csv')
print(df)
x=df[['radius_mean', 'texture_mean', 'smoothness_mean',
       'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
       'symmetry_se', 'fractal_dimension_se']]
y=df['diagnosis']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xtrain=ss.fit_transform(xtrain)
xtest=ss.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr=LogisticRegression()
lr.fit(xtrain,ytrain)

'''prediction1=model1.predict(xtest)
acc_log = accuracy_score(ytest,prediction1)*100
print('Accuracy of the model: {0}%'.format(acc_log))'''
pickle.dump(lr,open("model.pkl","wb"))
model=pickle.load(open('model.pkl','rb'))

