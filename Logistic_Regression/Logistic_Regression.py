import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


#导入数据
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5',"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5',"r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes


#初始化参数
def initialize_with_zeros(dim):
    """
    此函数为w创建一个形状为(dim,1)的零向量，并将b初始化为0。

    输入：
    dim -- w向量的大小

    输出：
    s -- 初始化的向量
    b -- 初始化的偏差
    """

    w = np.zeros((dim,1))
    b=0

    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b


#计算代价函数及其梯度
def propagate(w,b,X,Y):
    """
    实现前向传播的代价函数及反向传播的梯度

    输入：
    w -- 权重，一个numpy数组，大小为（图片长度*图片高度*3，1）
    b -- 偏差，一个标量
    X -- 训练数据，大小为（图片长度*图片高度*3，1）
    Y -- 真实“标签”向量，大小为（1，样本数量）

    输出：
    cost -- 逻辑回归的负对数似然代价函数
    dw -- 相对于w的损失梯度，因此与w的形状相同
    db -- 相对于b的损失梯度，因此与b的形状相同
    """
    
    m = X.shape[1]

    #前向传播
    Z = np.dot(w.T,X)+b
    A = 1/(1 + np.exp(-Z))
    cost=np.sum(np.multiply(Y,np.log(A))+np.multiply(1-Y,np.log(1-A)))/(-m)

    #反向传播
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw":dw,"db":db}

    return grads,cost


#使用优化算法（梯度下降法）
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    """
    此函数通过运行梯度下降算法来优化w和b

    输入：
    w -- 权重，一个numpy数组，大小为（图片长度*图片高度*3，1）
    b -- 偏差，一个标量
    X -- 训练数据，大小为（图片长度*图片高度*3，样本数量）
    Y -- 真实“标签”向量，大小为（1，样本数量）
    num_iterations -- 优化循环的迭代次数
    learning_rate -- 梯度下降更新规则的学习率
    print_cost -- 是否每100步打印一次丢失

    输出：
    params -- 存储权重w和偏差b的字典
    grads -- 存储权重梯度相对于代价函数偏倒数的字典
    costs -- 在优化期间计算的所有损失的列表，这将用于绘制学习曲线。
    """

    costs=[]

    for i in range(num_iterations):

        #成本的梯度计算
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        #更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i % 100 == 0:
            costs.append(cost)

        #每100次训练迭代打印成本
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i:%f" %(i,cost))

        params = {"w":w,"b":b}
        grads = {"dw":dw,"db":db}

    return params,grads,costs



#定义预测函数
def predict(w,b,X):
    """
    使用学习的逻辑回归参数(w,b)预测标签是0还是1

    输入：
    w -- 权重，一个numpy数组，大小为（图片长度*图片高度*3，1）
    b -- 偏差，一个标量
    X -- 训练数据，大小为（图片长度*图片高度*3，样本数量）

    输出：
    Y_prediction -- 包含X中示例的所有预测（0/1）的numpy数组（向量）
    """


    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    #计算向量"A"预测猫出现在图片中的概率
    A = 1 / (1 + np.exp(-(np.dot(w.T,X)+b)))

    #将概率A[0,i]转换为实际预测p[0,i]
    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
        pass

    assert(Y_prediction.shape == (1,m))

    return Y_prediction




#定义模型训练函数
def model(X_train,Y_train,X_test,Y_test,num_iterations = 200,learning_rate = 0.5,print_cost = False):
    """
    通过调用前面实现的函数来构建逻辑回归模型

    输入：
    X_train -- 由numpy数组表示的训练集，大小为（图片长度*图片高度*3，训练样本数）
    Y_train -- 由numpy数组（向量）表示的训练标签，大小为（1，训练样本数）
    X_test -- 由numpy数组表示的测试集，大小为（图片长度*图片高度*3，测试样本数）
    Y_test -- 由numpy数组（向量）表示的测试标签，大小为（1，测试样本数量）
    num_iterations -- 超参数，表示优化参数的迭代次数
    learning_rate -- 超参数，在优化算法更新规则中使用的学习率
    print_cost -- 设置为True时，以每次100次迭代打印成本

    输出：
    d -- 包含模型信息的字典。
    """

    #初始化参数
    w,b = initialize_with_zeros(X_train.shape[0])

    #梯度下降
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    #从字典"parameters"中检索参数w和b
    w = parameters["w"]
    b = parameters["b"]

    #预测测试/训练集
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    #打印训练/测试集的预测准确率
    print("train accuracy:{} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy:{} %".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d = {"costs":costs,"Y_prediction_test":Y_prediction_test,"Y_prediction_train":Y_prediction_train,"w":w,"b":b,"learning_rate":learning_rate,"num_iterations":num_iterations}

    return d




#数据加载
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

#将数据集转换为矢量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#数据标准化
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


#模型训练
d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 2000,learning_rate = 0.005,print_cost = True)


#测试
num_px = train_set_x_orig.shape[1]
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px,num_px,3)))
print("y = " + str(test_set_y[0,index]) + ",you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") + "\" picture.")


#绘制学习曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


#学习率的选择
#将模型的学习曲线与几种学习率进行比较
learning_rates = [0.01,0.001,0.0001]
models = {}
for i in learning_rates:
    print("Learning rate is: " + str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 1500,learning_rate = i,print_cost = False)
    print('\n'+'----------------------------------------------------------------------------------' + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))
    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc = 'upper center',shadow = True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()




#使用自己的图像进行测试
def test(test_img):
    img = np.array(ndimage.imread(test_img,flatten = False))
    my_img = scipy.misc.imresize(img,size = (num_px,num_px)).reshape((1,num_px*num_px*3)).T
    my_predicted_img = predict(d["w"],d["b"],my_img)

    plt.imshow(img)
    print("y = " + str(np.squeeze(my_predicted_img)) + ",your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_img)),].decode("utf-8") + "\" picture.")
    
        
