# Logistic Regression performed quite well but perhaps random forest will perform much better
# Upload the pokemon and digimon images
import os
import sys
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def read_img_data(img_path,X,j):
	# Reads the image data in img_path and starts placing the data in the X matrix
	filenames = os.listdir(img_path)
	print("Uploading Images ...")
	for i in range(len(filenames)):
	    file = filenames[i]
	    image = Image.open(img_path+"\\"+file).convert("RGB")
	    data = np.asarray(image).flatten()
	    X[i+j,:] = data

	print("Images successfully uploaded!")

	return X,len(filenames)

def create_X_y(poke_path,digi_path):
	# Create an empty x matrix where rows = no. of poke images+no. of digi images and cols = no. of pixels in each rgb layer
	poke_filenames = os.listdir(poke_path)
	digi_filenames = os.listdir(digi_path)

	# Finding the dimensions of an image
	test_img = Image.open(os.path.join(poke_path,poke_filenames[0]))
	dims = np.shape(test_img)

	pixel_no = dims[0]*dims[1]*3
	poke_img_no = len(poke_filenames)
	digi_img_no = len(digi_filenames)

	X = np.zeros(shape=(poke_img_no+digi_img_no,pixel_no))
	
	# Read/upload data in to the matrix
	X,j = read_img_data(poke_path,X,0)
	X,j = read_img_data(digi_path,X,j)

	# Pokemon = 0, and Digimon = 1
	y = np.zeros(shape=(poke_img_no+digi_img_no,))
	y[poke_img_no:] = np.ones((digi_img_no,))

	return X,y

def random_fores(X,y):
	X = preprocessing.scale(X)

	# Split the dataset into training, test and validation set.
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
	X_train,X_val,y_train,y_val= train_test_split(X_train,y_train,test_size=0.2,random_state=1)

	# Train the model with logistic regression
	print("\nTraining the random forest model ...")

	best_model = RandomForestClassifier(random_state=1).fit(X_train,y_train)
	y_pred = best_model.predict(X_val)

	return y_val,y_pred,best_model,X_val


def results(y_val,y_pred,best_model,X_val):
	# Returns the accuracy, recall and precision of the prediction
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
	plt.plot(fpr,tpr,label="default auc="+str(auc))
	plt.legend(loc=4)
	print("AUC:",auc)
	plt.show()


def main():
	if len(sys.argv)!=3:
		print("Warning: Incorrect Usage. Example: python random_forest_parameter_tuning.py poke_path digi_path")
		return

	poke_path = sys.argv[1]
	digi_path = sys.argv[2]

	# Uploading image data to create the X matrix
	X,y = create_X_y(poke_path,digi_path)

	# Train the machine learning model and check its accuracy with a validation set
	y_val,y_pred,best_model,X_val = random_fores(X,y)
	results(y_val,y_pred,best_model,X_val)

if __name__ == "__main__":
    main()