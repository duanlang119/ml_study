from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
data=[[0,0,3],[1,1,0],[0,2,1],[1,0,2]]
print(data)

enc.fit(data)

x=[[0,1,3]]
print('再来看要进行编码的参数：')
print(x)
print('onehot编码的结果：')
print(enc.transform(x).toarray())
