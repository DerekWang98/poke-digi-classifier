# Use logistic_regression and regularisation

# Upload the pokemon and digimon images
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

poke_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_png_small"
digi_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_png_small"

poke_filenames = os.listdir(poke_path)
digi_filenames = os.listdir(digi_path)

pixel_no = 215*215*3
poke_img_no = 809
digi_img_no = 1284


X = np.zeros(shape=(poke_img_no+digi_img_no,pixel_no))

print("Uploading Pokemon Images ...")
for i in range(len(poke_filenames)):
    file_name = poke_filenames[i]
    image = Image.open(poke_path+"\\"+file_name).convert("RGB")
    data = np.asarray(image).flatten()
    X[i,:] = data

print("Images successfully uploaded!")

print("Uploading Digimon Images ...")
for i in range(len(digi_filenames)):
    file_name = digi_filenames[i]
    image = Image.open(digi_path+"\\"+file_name).convert("RGB")
    data = np.asarray(image).flatten()
    X[i,:] = data

print("Images successfully uploaded!")

# Pokemon = 0, and Digimon = 1
y = np.zeros(shape=(poke_img_no+digi_img_no,))
y[poke_img_no:] = np.ones((digi_img_no,))

X = preprocessing.scale(X)

# Split the dataset into training, test and validation set.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train,X_val,y_train,y_val= train_test_split(X_train,y_train,test_size=0.2,random_state=1)

print("Hyper parameter tuning ... \n")
model = LogisticRegression(max_iter=1000,random_state=1)
penalty=['l1','l2']
C = np.logspace(-4,-2,50)
hyperparameters = dict(C=C,penalty=penalty,solver=['liblinear'])
clf = GridSearchCV(model, hyperparameters, cv = 5, verbose=0)
best_model = clf.fit(X_train, y_train)
y_pred = best_model.predict(X_val)
print("Best penalty:",best_model.best_estimator_.get_params()['penalty'])
print("Best C:",best_model.best_estimator_.get_params()['C'])

cnf_matrix = metrics.confusion_matrix(y_val, y_pred)
TP,FN,FP,TN = cnf_matrix[0,0],cnf_matrix[0,1],cnf_matrix[1,0],cnf_matrix[1,1]

acc = (TP+TN)/(TP+TN+FP+FN)
rec = TP/(TP+FN)
pre = TP/(TP+FP)
print("Accuracy:",acc)
print("Recall:",rec)
print("Precision",pre)

y_pred_proba = best_model.predict_proba(X_val)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_val,  y_pred_proba)
auc = metrics.roc_auc_score(y_val, y_pred_proba)