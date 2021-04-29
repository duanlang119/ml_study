import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#特征文字

feat_cols =['mass','width','height','color_score']

#读取数据

data = pd.read_csv('fruit_data.csv')

#预处理

fruit2num = {'apple'    :  0 ,
             'mandarin' :  1 ,
             'orange'   :  2 ,
             'lemon'    :  3
             }
data['label'] = data['fruit_name'].map(fruit2num)
#取出X和y

X = data[feat_cols].values
y = data['label'].values

#划分数据

X_train_set, X_test_set ,y_train_set, y_test_set = train_test_split(X,y, random_state = 20, test_size= 1/5)

print('原始数据集共{}个样本，其中训练集样本数为{}，测试集样本数为{}'.format(
    X.shape[0], X_train_set.shape[0], X_test_set.shape[0]))

#训练

knn_model = KNeighborsClassifier(n_neighbors=5,weights='distance',p=2)

knn_model.fit(X_train_set, y_train_set)

#准确率检测

accur = knn_model.score(X_test_set,y_test_set)

print(f'正确率为{accur*100}%')

#试试看

num2fruit = dict(zip(fruit2num.values(),fruit2num.keys()))

for idx in range(X_test_set.shape[0]):
    test_feat = [X_test_set[idx]]
    y_pridict = num2fruit.get(int(knn_model.predict(test_feat)))
    y_real = num2fruit.get(y_test_set[idx])
    YorN = '对' if y_pridict == y_real else '错'
    print(f'第{idx+1}个测试水果的结果是{y_pridict}，本来应该是{y_real}，所以测{YorN}了')