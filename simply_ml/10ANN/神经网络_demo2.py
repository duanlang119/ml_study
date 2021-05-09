# pip install scikit-learn --upgrade
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
#载入数据
mnist = fetch_mldata('MNIST original',data_home='c://pythoncode//scikit_learn_data')
print('样本数量：{},样本特征数：{}'.format(mnist.data.shape[0],mnist.data.shape[1]))

X  = mnist.data/255 #数据
y = mnist.target #标签
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=20000,
                                                 test_size=1000,random_state=62) #分割数据1/4为测试数据，3/4为训练数据
mlp_hw=MLPClassifier (solver='adam',hidden_layer_sizes=[100,100,100],
                      activation='relu',random_state=62)
mlp_hw.fit(X_train,y_train)
print('代码运行结果')
print('测试数据集得分: {:.2f}%'.format(mlp_hw.score(X_test,y_test)*100))
from  PIL import Image
import numpy as np


def convertpic(filename):
    image=Image.open(filename).convert('F')
    image=image.resize((28,28))
    arr=[]
    for i in range(28):
        for j in range(28):
            pixel=1.0-float(image.getpixel((j,i)))/255.
            arr.append(pixel)
    return(np.array(arr).reshape(1,-1))

for i in range(10):
    filename = 'c:\\pythoncode\\test_pic\\'+str(i)+'.jpg'
    arr=convertpic(filename)
    print('图片文件名: {},图片中的数字是：{:.0f}'.format(str(i)+'.jpg'
                                            ,mlp_hw.predict(arr)[0]))

