# Logistic Regression performed quite well but perhaps random forest will perform much better
# Upload the pokemon and digimon images
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
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

pixel_no = 40*40*3
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
    X[i+len(poke_filenames),:] = data

print("Images successfully uploaded!")

# Pokemon = 0, and Digimon = 1
y = np.zeros(shape=(poke_img_no+digi_img_no,))
y[poke_img_no:] = np.ones((digi_img_no,))

X = preprocessing.scale(X)

# Split the dataset into training, test and validation set.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train,X_val,y_train,y_val= train_test_split(X_train,y_train,test_size=0.2,random_state=1)

# Train the model with logistic regression
print("Training the random forest model ... \n")

# best_model = RandomForestClassifier(n_estimators=100,random_state=1).fit(X_train,y_train)
# y_pred = best_model.predict(X_val)

# hyperparameters = {'bootstrap': [True, False],
#  'max_depth': [10, 50, 100],
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [10, 20, 40]}

# hyperparameters = {"n_estimators":[100,1000,10000],'max_depth': [10, 100, 1000],'max_features': ['auto', 'sqrt'],'min_samples_leaf': [10, 100, 1000]}
best_model = RandomForestClassifier(random_state=1).fit(X_train,y_train)
y_pred = best_model.predict(X_val)

# model = RandomForestClassifier(random_state=1)
# clf = GridSearchCV(model, hyperparameters, cv = 5, verbose=1)
# best_model = clf.fit(X_train, y_train)
# y_pred = best_model.predict(X_val)

# print("bootstrap:",best_model.best_model.best_estimator.get_params()["bootstrap"])
# print("n_estimators:",best_model.best_estimator_.get_params()["n_estimators"])
# print("max_depth:",best_model.best_estimator_.get_params()["max_depth"])
# print("max_features:",best_model.best_estimator_.get_params()["max_features"])
# print("min_samples_leaf:",best_model.best_estimator_.get_params()["min_samples_leaf"])

# print(best_model.get_params())

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
print("AUC:",auc)

# plt.plot(fpr,tpr,label="default auc="+str(auc))
# plt.legend(loc=4)
# plt.show()