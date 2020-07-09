# What is this project?
I've been learning a great deal of machine learning in my summer research scholarship at the Westmead Institute of Medical Research (WIMR) where I'm developing a radiomics tool for clinicians. In short, it's an application that will utilise machine learning to predict certain clinical outcome such as whether cancer will spread after treatment or is a tissue cancerous or benign. These predictions hopefully will help doctors by providing additional insights using past CT data.
I want to consolidate my learnings with a fun personal project and what better way than to make a Pokemon and Digimon classifier. People need to be able to understand that Pokemons are different to Digimons >:)

TL;DR I wanted a fun project to summarise what I've learnt so far in my summer research scholarship.

# Where did I get the images from?
I downloaded the image of every Digimon from https://wikimon.net/Visual_List_of_Digimon and every Pokemon (up to Generation 7/Sun and Moon) from https://www.pokemon.com/us/pokedex/ using a chrome extension https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en.

# What did I do (in summary)?
## 1) Filter useless images
I made sure to filter out useless images such as logos/banners that might have been in the website since the chrome extensions downloads all images on a website. 
## 2) Resized images to a consistent size
Converted them into PNG so that a python package, resizeimage, will be able to make them into a consistent pixel size (I chose 215 x 215 pixels since most pokemon images were already that size). Converted them into numbers into a numpy array, did some feature scaling and fed them into a logistic regression model with ridge regularisation. 
```shell
It ended up having an accuracy of 0.8 and an AUC of 0.878 on the validation set.
```
Area under the ROC (receiver operator curve) curve (AUC) highlights how well the model is doing. An AUC of 1 means all examples were classified correctly and 0 means everything were incorrectly classified. 
## 3) Grid search (hyperparameterisation)
A grid search was performed which finds the best parameter combination for the model. However, the large size of the images made it it difficult to  and found that the large datasets take significantly more time so I resized the images to 40 by 40 pixels. The grid search returned values of penalty = 'l2' and C= 0.0020235896477251557.
```shell
This gave an accuracy of 0.8 and AUC of 0.877 which did not really improve at all.
```
## 4) Refactor code
In the beginning of the project, I was very excited about doing the project and neglected code quality. I decided to refactor the code after a few months later which I found mistakes within the code and magic numbers everywhere. Fixing all these up, I realised the data was never fed into the models properly (I replaced pokemond data with digimon data rather than adding them both to the matrix).
### Logistic Regression:
```shell
Accuracy: 0.96
Recall: 0.94
Precision 0.98
AUC: 0.99
This is an amazing result!! But can we do better?
```
### Logistic Regression with hyperparameters:
```shell
Hyperparameters: [Best penalty: l1 and Best C: 0.0020235896477251557]
Accuracy: 0.991044776119403
Recall: 1.0
Precision 0.9794520547945206
AUC: 1.0
The model could be overfitted. 10-fold or k-fold cross validation will need to be performed with these parameters to check.
```

### Random Forest:
```shell
Accuracy: 0.9791044776119403
Recall: 0.958041958041958
Precision 0.9927536231884058
AUC: 0.995465472027972
Random forest gave slightly better results than logistic regression which is expected. 
```

### Random Forest with hyperparameters:
```shell
Hyperparameters: [bootstrap: True, n_estimators: 100, max_depth: 10, max_features: auto, min_samples_leaf: 10]
Accuracy: 1.0
Recall: 1.0
Precision 1.0
AUC: 1.0
The model could be overfitted. 10-fold or k-fold cross validation will need to be performed with these parameters to check.
```

### Logistic Regression:
# What I've learnt and future steps?
1) During the summer research scholarship, it helped me familiarise myself with machine learning packages such as scikitlearn by reading documentation and also how to deal with image data.

2) The value of refactoring code or making sure code quality is good to begin. I was pleasantly surpised at the accuracy of the model after fixing the data reading issue. However, more work can be done to ensure that the model is not overfitting. 10-fold or k-fold cross validtion can be used to check for overfitting.
