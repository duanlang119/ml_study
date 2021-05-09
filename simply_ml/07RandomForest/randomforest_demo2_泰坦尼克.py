import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

data_file_path="titanic3.xls"
df_data=pd.read_excel(data_file_path)
selected_cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
selected_df_data=df_data[selected_cols]

#数据处理
def prepare_data(df_data):
    df=df_data.drop(['name'],axis=1)#名字训练时不需要，去掉
    age_mean=df['age'].mean()
    df['age']=df['age'].fillna(age_mean)#缺失的年龄以平均值填充
    fare_mean=df['fare'].mean()
    df['fare']=df['fare'].fillna(fare_mean)#缺失的票价以平均值填充 
    df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)#文字转化为数字表示
    df['embarked']=df['embarked'].fillna('S')#缺失值用最多的值取代
    df['embarked']=df['embarked'].map({'C':0,'Q':1,'S':2}).astype(int)#文字转化为数字表示
    ndarray_data=df.values
    features=ndarray_data[:,1:]#没有生存情况
    label=ndarray_data[:,0]#生存情况
    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    norm_features=minmax_scale.fit_transform(features)#归一化
    return norm_features,label
shuffled_df_data=selected_df_data.sample(frac=1)#打乱顺序
x_data,y_data=prepare_data(shuffled_df_data)
train_size=int(len(x_data)*0.8)#80%的数据训练，20%的数据测试
x_train=x_data[:train_size]#训练数集
y_train=y_data[:train_size]
x_test=x_data[train_size:]#测试数集
y_test=y_data[train_size:]

srfc = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=10)
srfc.fit(x_train, y_train)
# 准确度

print("分类得分：",srfc.score(x_test, y_test))
from sklearn.metrics import classification_report
y_pre=srfc.predict(x_test)
print(classification_report(y_test,y_pre))
#预测
Jack_infor=[0,'Jack',3,'male',23,1,0,5.000,'S']
Rose_infor=[1,'Rose',1,'female',20,1,0,100.000,'S']
new_passenger_pd=pd.DataFrame([Jack_infor,Rose_infor],columns=selected_cols)#创建新旅客的表单
all_passenger_pd=selected_df_data.append(new_passenger_pd)#与旧的合成
x,y=prepare_data(all_passenger_pd)
y_pre=srfc.predict(x[-2:,:])
print("Jack与Rose，")
for i in range(len(y_pre)):
    print("生存：",y_pre[i])




