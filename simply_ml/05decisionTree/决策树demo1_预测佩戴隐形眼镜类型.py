 #############################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate','class']#特征标签+类别标签
feature = ['age', 'prescript', 'astigmatic', 'tearRate']#特征标签
filename='lenses.txt'
lenses = pd.read_table(filename, names=lensesLabels, sep='\t')
#names：设置列名 ，sep:分隔的正则表达式,'/t'表示以tab键进行分割
def main():
    x_train,y_train = lenses[feature],lenses['class']  # 取出特征数据和特征结果
    le = LabelEncoder()
    # 创建LabelEncoder()对象，用于序列化
    for col in x_train.columns:  # 分列序列化(给字符编写数字编号)
        x_train[col] = le.fit_transform(x_train[col])
    x_train=x_train.values
    #将数据分为训练数据和测试数据
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)
    # 创建决策树对象
    clf = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=4)
    #根据数据，构造决策树
    model = clf.fit(x_train, y_train)
    # 预测
    y_pred = model.predict(x_test)
    # 输出
    print("正确值：\n{0}".format(y_test))
    print("预测值：\n{0}".format(y_pred))
    print("准确率：%f%%" % (accuracy_score(y_test, y_pred) * 100))
if __name__ == '__main__':
    main()
###########################################
