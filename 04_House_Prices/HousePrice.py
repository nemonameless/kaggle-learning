import pandas as pd
import numpy as np

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],test.loc[:, 'MSSubClass':'SaleCondition']))
#train.head()
#train.shape
#test.shape

# 使用pandas对数值特征做标准化处理：
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
# 把离散数据点转换成数值标签
all_X = pd.get_dummies(all_X, dummy_na=True)
# 缺失数据用本特征的平均值估计
all_X = all_X.fillna(all_X.mean())
# 数据转换一下格式
num_train = train.shape[0]
X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()

# 便于和Gluon交互，需要导入NDArray格式数据
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))
X_test = nd.array(X_test)
# 损失函数定义为平方误差
square_loss = gluon.loss.L2Loss()

# 定义比赛中测量结果用的函数
def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)

# 模型的定义，是一个基本的线性回归模型？
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1024,activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

# %matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs, verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 128
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', 
    						{'learning_rate': learning_rate, 
    						 'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)

    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:    
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    '''
    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train','test'])
    plt.show()
    '''
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss

# 过度依赖训练数据集的误差来推断 测试数据集的误差 容易导致过拟合。
# 当调参时往往需要基于K折交叉验证，把初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K−1个样本用来训练
# 我们关心K次 验证模型的测试结果的平均值和 训练误差的平均值
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        # K折的训练
        train_loss, test_loss = train(net, X_val_train, y_val_train, X_val_test, y_val_test, 
                                      epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

k = 5
epochs = 50
verbose_epoch = 45
learning_rate = 0.05
weight_decay = 250

# train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay)
# print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" % (k, train_loss, test_loss))

# 定义预测函数
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay):
    net = get_net()
    # 训练，只返回cur_train_loss
    train(net, X_train, y_train, None, None, 
    	  epochs, verbose_epoch, learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('hp_wd250.csv', index=False)
    
learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay)