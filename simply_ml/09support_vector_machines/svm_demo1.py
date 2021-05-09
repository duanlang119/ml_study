from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
boston=load_boston()
X, y = boston.data, boston.target
X_train, X_test,y_train,y_test=train_test_split(X,y,random_state=8)
print(X_train.shape)
print(X_test.shape)
from sklearn.svm import SVR
for kernel in ['linear','rbf']:
    svr=SVR(kernel=kernel)
    svr.fit(X_train,y_train)
    print(kernel,'核函数的模型测试集得分: {:.3f}'.format(svr.score(X_test,y_test)))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scalerd=scaler.transform(X_train)
X_test_scalerd=scaler.transform(X_test)
for kernel in ['linear','rbf']:
    svr=SVR(kernel=kernel,C=100,gamma=0.1)
    svr.fit(X_train_scalerd,y_train)
    print('数据进行预处理后，并且调整了gamma和C两个参数')
    print(kernel,'核函数的模型测试集得分: {:.3f}'.format(svr.score(X_test_scalerd,y_test)))

