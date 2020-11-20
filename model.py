import pandas as pd
from sklearn.datasets import load_wine
wine = load_wine()
from sklearn.model_selection import train_test_split
from sklearn import svm
classifier = svm.SVC(kernel = "linear")
import pickle

features = pd.DataFrame(data=wine["data"],columns=wine["feature_names"])

features["target"] = wine["target"]
features['class'] = features['target'].map(lambda ind: wine['target_names'][ind])

data_train, data_test, label_train, label_test = train_test_split(wine['data'],wine['target'],test_size=0.25)

classifier.fit(data_train,label_train)

pickle.dump(classifier, open('model.pkl','wb'))