import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

print("###########缺失值补全###################")
# imp = SimpleImputer(missing_values='NaN',strategy='mean',axis=0)
imp = SimpleImputer(missing_values=np.nan,strategy='mean')
#训练模型，拟合作为替换值的均值
# imp.fit([[1,2],[np.nan,3],[7,6]])

data =[[np.nan,2],[6,np.nan],[7,6]]
imp.fit(data)

#这里SimpleImputer仅仅只是计算了每个属性的中位数的值，并将结果存储到该类的实例变量statistics_中：
print(imp.statistics_)

print(imp.transform(data))

data2=["Japan","China","Japan","Korea","China"]
print(data2)
le = LabelEncoder()
le.fit(data2)
print('标签个数：%s' % le.classes_)
print('标签值标准化：%s' % le.transform(data2))
data3=["Japan","China","China","Korea","Korea"]
print(data3)
print('标签值标准化：%s' % le.transform(data3))