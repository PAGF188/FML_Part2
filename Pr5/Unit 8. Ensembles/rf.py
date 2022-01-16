# random forest using a training and a testing set
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.model_selection import *
from numpy import *
# 3 classes
#dataset='wine';x=loadtxt('wine.data');
# 2 classes
dataset='hepatitis';x=loadtxt('hepatitis.data');
c=x[:,0]-1 # the true vulue
x=delete(x,0,1) # delete column 0
[N,I]=x.shape
cl=unique(c);C=len(cl);
# preprocessing: mean 0, desviation 1
x=(x-mean(x,0))/std(x,0)
# Split the dataset in two equal parts
Xtrain, Xtest, ytrain, ytest = train_test_split(x, c, test_size=0.5, random_state=0)
# Create the RF classifier with 10 random trees
model = RandomForestClassifier(n_estimators=10)
# Train the classifier
model = model.fit(Xtrain, ytrain)
# Compute the prediction of the classifier for the test set
y=model.predict(Xtest)
kappa=cohen_kappa_score(ytest, y)
print('Dataset=%s, kappa=%.2f%%'%(dataset,kappa*100))
cf=confusion_matrix(ytest,y)
print('Confusion matrix'); print(cf)
a=accuracy_score(ytest, y)
print('Accuracy= %.2f'%a)
pre=precision_score(ytest,y)
re=recall_score(ytest,y)
f1=f1_score(ytest,y)
print('Precision=%.2f, recall=%.2f and F1=%.2f' %(pre, re, f1))
# ROC curve ----------------------------------
aux=model.predict_proba(Xtest)
p=aux[:,1] # probability of class 1
fpr, tpr, thresholds = roc_curve(ytest,p)
from matplotlib.pyplot import *
clf(); plot(fpr,tpr,'bs--'); 
ylabel('True positive rate') 
xlabel('False positive rate')
title('AUC= %.4f'% roc_auc_score(ytest,p))
grid(True); show(False)

