import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
# DataFrame:
df=pd.DataFrame(iris.data,columns=iris.feature_names)
# 前五行
print(df[:5])
iris_x=iris.data
iris_y=iris.target
# 前两行分类应该是0
print(iris_y[:5])
print('数据集规格大小',iris_x.shape)
print('标签有',iris.target_names)
print('数据的属性有',iris.feature_names)

X_train,X_test,Y_train,Y_test=train_test_split(iris_x,iris_y,test_size=0.3,random_state=42)

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,Y_train)

per=clf.predict(X_test)
print('真实值',Y_test)
print('预测值',per)

print(f'Gaussian Train Score: {clf.score(X_train,Y_train)}')
print(f'Gaussian Test Score: {clf.score(X_test,Y_test)}')

from sklearn.naive_bayes import MultinomialNB
clf1=MultinomialNB()
clf1.fit(X_train,Y_train)

per1=clf1.predict(X_test)
print('真实值',Y_test)
print('预测值',per1)

print(f'MultinomialNB Train Score: {clf1.score(X_train,Y_train)}')
print(f'MultinomialNB Test Score: {clf1.score(X_test,Y_test)}')

from sklearn.naive_bayes import BernoulliNB
clf2=BernoulliNB()
clf2.fit(X_train,Y_train)

per2=clf1.predict(X_test)
print('真实值',Y_test)
print('预测值',per2)

print(f'BernoulliNB Train Score: {clf2.score(X_train,Y_train)}')
print(f'BernoulliNB Test Score: {clf2.score(X_test,Y_test)}')

