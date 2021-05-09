from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_wine 
wine = load_wine()
X=wine.data
y=wine.target
print(wine.data)
print(wine.data.shape)
print(type(X))
# 特征列名和标签分类
print(wine.feature_names)
print(wine.target_names)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.4)
#对比随机森林分类器vs决策树
clf = DecisionTreeClassifier(random_state=1,criterion='entropy')
rfc = RandomForestClassifier(random_state=1,criterion='entropy',n_estimators=25)
clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)
score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)
print("Single Tree:{}".format(score_c)
      ,"Random Forest:{}".format(score_r)
     )

#为了观察更稳定的结果，下面进行十组交叉验证    
rfc_l = []
clf_l = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc,X,y,cv=10).mean()
    rfc_l.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf,X,y,cv=10).mean()
    clf_l.append(clf_s)
    
plt.plot(range(1,11),rfc_l,label = "Random Forest")
plt.plot(range(1,11),clf_l,label = "Decision Tree")
plt.legend()
plt.show()
