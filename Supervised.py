#Some Sample Datasets
# generate dataset
import mglearn as mg
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
X,y=mg.datasets.make_forge()
# plot dataset
mg.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["birinci","İkinci"],loc=4)
plt.xlabel("dikey veri")
plt.ylabel("yatay veri")
print(X)
plt.show()

X2,y2=mg.datasets.make_wave()
plt.plot(X2,y2,'o')
plt.ylim(-3,3)
plt.xlabel("feature")
plt.ylabel("Target")
plt.show();
#cancer
from  sklearn.datasets import load_breast_cancer 
cancer=load_breast_cancer()
print("cancer keys:\n {}".format(cancer.keys())) #bunun yerine cabcer["key"] kullanılabilir
print("shape of cancer data: {}".format(cancer.data.shape))
print("sample counts per class:\n{}".format({n:v for n ,v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("Feature names: {}".format(cancer.feature_names))
#houses
from mglearn.datasets import load_extended_boston
X3,y3=load_extended_boston()
print("X.shape \n {}".format(X3.shape))
#k-Nearest Neighbors //en yakınddaki noktalar arası en kısa mesafeyi bulan algoritmadır
import mglearn.plots  as knn
#knn.plot_knn_classification(n_neighbors=1) //en yakın 1 komsuyu seçer dafa fazlası da seçilebilir
#example knn app
from sklearn.model_selection import train_test_split
import mglearn as mg2
X4,y4=mg2.datasets.make_forge()
X_train,X_test,Y_train,Y_test=train_test_split(X4,y4,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,Y_train)
print("test set predictions: {}".format(clf.predict(X_test)))
print("test : {}".format(clf.score(X_test,Y_test)))
fig,axes=plt.subplots(1,3,figsize=(10,4))
for n_neighbors,ax in zip([1,3,5,9],axes):
  clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X4,y4)
  mg2.plots.plot_2d_separator(clf,X4,fill=True,eps=0.5,ax=ax,alpha=.4)
  mg2.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title("{} neighbor(s)".format(n_neighbors))
  ax.set_xlabel("feature 0")
  ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
#başka örnek
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X5_train,X5_test,Y5_train,Y5_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)
training_accuracy = []
test_accuracy = []
#try n_neighbors from 1 to 10
neighbors_setting=range(1,11)
for n_neighbors in neighbors_setting:
    #build the model
    clf2=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf2.fit(X5_train,Y5_train)
    # record training set accuracy
    training_accuracy.append(clf2.score(X5_train,Y5_train))
    test_accuracy.append(clf2.score(X5_test,Y5_test))

plt.plot(neighbors_setting,training_accuracy,label="training accuracy")
plt.plot(neighbors_setting,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#k-neighbors regression
mg2.plots.plot_knn_regression(n_neighbors=1)
plt.show()
mg2.plots.plot_knn_regression(n_neighbors=3)
plt.show()







