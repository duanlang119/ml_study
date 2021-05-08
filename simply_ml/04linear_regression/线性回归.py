import numpy as np

#我们需要定义数据集和学习率
#数据集大小，20个数据点
m=20
#x的坐标以及对应的矩阵
X0 = np.ones((m,1))
X1 = np.arange(1,m+1).reshape(m,1)
X = np.hstack((X0,X1))
# 对应的Y坐标
Y=np.array([3,4,5,5,2,4,7,8,11,8,12,11,13,13,16,17,18,17,19,21,]).reshape(m,1)
# 学习率
alpha = 0.01
print(X)

# 我们以矩阵向量的形式定义代价函数和代价函数的梯度
# 定义代价函数
def cost_function(theta,X,Y):
    diff = np.dot(X,theta) - Y
    return (1/(2*m)) * np.dot(diff.transpose(),diff)

# 定义代价函数对应的梯度函数
def gradient_function(theta,X,Y):
    diff = np.dot(X, theta) - Y
    return (1/m) * np.dot(diff.transpose(), diff)

# 算法的核心部分，梯度下降迭代计算
# 梯度下降迭代
def gradient_descent(X,Y,alpha):
    theta = np.array([1,1]).reshape(2,1)
    gradient = gradient_function(theta,X,Y)

    while not all(abs(gradient)<=1e-5):
        theta = theta - alpha*gradient
        gradient = gradient_function(theta,X,Y)
    return theta

optimal = gradient_descent(X,Y,alpha)
print('optimal',optimal)
print('cost function',cost_function(optimal,X,Y)[0][0])

# 画出对应的图像
def plot(X,Y,theta):
    import matplotlib.pyplot as plt
    ax = plt.subplots(111)
    ax.scatter(X,Y,s=30,c='blue')
    plt.xlabel("X")
    plt.ylabel("Y")
    x = np.arange(0,21,0.2)
    y = theta[0]+theta[1]*x
    ax.plot(x,y)
    plt.show()

plot(X1,Y,optimal)



