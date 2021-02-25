from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

win = load_wine()
print(win.data.shape)
X_train,X_test,y_train,y_test = train_test_split(win.data,win.target,random_state=62)

print(X_train.shape,X_test.shape)


mlp=MLPClassifier(hidden_layer_sizes=[100,100],max_iter=400,random_state=62)
mlp.fit(X_train,y_train)

print(f'没有处理的情况下： {mlp.score(X_test,y_test)}')

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_pp=scaler.transform(X_train)
X_test_pp=scaler.transform(X_test)

mlp.fit(X_train_pp,y_train)

print(f'有处理的情况下： {mlp.score(X_test_pp,y_test)}')



