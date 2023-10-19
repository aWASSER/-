# 机器学习的分类 三大要素: 数据 算法 算力
"""
有监督学习: 由已知的样本作为训练集, 建立数学模型, 之后预测未来样本, 特征(x)与标签化(y)训练数据集中推断出模型等等机器学习任务
无监督学习: 训练集中没有人为标注(y)的结果, 数据并不被特别标识, 目的:为了推断出数据的一些内在结构
半监督学习: 用少量样本和大量未标注样本进行训练分类的问题
"""

# 有监督学习
"""
1. 分类: 离散
    判别式模型 输入x 判断y的值: 直接对条件概率建模: logistic回归, 决策树, 支持向量机SVM, k邻近, 神经网络
    生成式模型 输入x 判断y出现的值哪个概率更大: 联合概率建模: 马尔科夫模型HMM, 朴素贝叶斯, 高斯混合模型 GMM, LDA等
2. 回归: 连续
"""

# 无监督学习
"""
提取数据背后的数据特征, 提取有用特征信息, 方法: 聚类, 降维, 特征抽取
需要有监督学习的前期数据处理
"""

# 半监督学习
"""
提高机器学习性能
模型: 平滑假设, 聚类假设, 流行假设
算法: 半监督分类, 半监督回归, 半监督聚类, 半监督降维
缺点: 抗干扰能力弱, 目前仅适用于实验室环境
"""

# 数据清洗和转化
"""
数据过滤: 
pd.read_excel/csv(路径)

数据缺失 -- 1. 删除  2. 填充 (均值 众数 中位数 常数) 连续均值 离散众数 
        1. df.dropna(axis="columns", how='all', inplace=True)
        2. studf.fillna({"分数":0}) or studf.loc[:, '分数'] = studf['分数'].fillna(0)
           studf.loc[:, '姓名'] = studf['姓名'].fillna(method="ffill")

异常值, 值错误 -- 1. 删除 2. 换行换列  3. 修改单位

合并
        1. pd.concat([df1,df2], axis=0, ignore_index = False, join = "outer")  
        2. df1.append(df2)

汇总:
    1. 清洗时一般使用1-of-k, 哑编码
    2. 文本中提取 词袋法 或 TF-IDF
    3. 傅里叶变换
    4. 正则化, 标准化, 使不同模型输入变量取值范围相同
    5. 数据类型转换
    6. 不断尝试构成虚拟变量
"""

# 模型训练以及测试
"""
训练: 运用算法, 迭代对数据进行修改, 被称为交叉验证(使用训练集构建模型, 用测试集评估模型并提出建议)
测试指标: 准确率/召回率/精准率/F值(F1指标)
    准确率(Accuracy) = 提取出样本正确个数 / 总样本数 
    召回率(Recall) = 正确正例样本数 / 样本中正例样本数 --覆盖率
    精准率(Precision) = 正确正例样本数 / 预测为正例样本数
    F值 = Precision * Recall * 2 / (Precision + Recall)   是 re 和 pr 的调和平均

                        预测值
                 正例           负例
真实值   正例    真正例(tp)A    假负例(fn)B
         负例    假正例(fp)C    真负例(tp)D

测试结果计算:
    Accuracy = (A + D) / (A + B + C + D)
    Recall = (A) / (A + B)
    Precision = (A) / (A + C)
    F = (2 * (A) / (A + C) * (A) / (A + B) ) / ((A) / (A + B) + (A) / (A + C))
"""

# 混淆矩阵
"""
                        预测值  
                 正例           负例                
真实值   正例    真正例A        假负例B         Recall = (A) / (A + B)
         负例    假正例C        真负例D         FPR
        Acc        Precision    FOR
"""

# ROC
"""
描述分类混淆矩阵中FPR 和 TPR 两个量之间的对比变化, 纵轴是真正例率 (TPR), 横轴是假正例率(FPR)

ROC 曲线描绘了FPR 和 TPR 之间权衡的情况, 通俗来说, 是看TPR 和 FPR 哪一个长得快, 快多少
    TPR 增长快: 曲线上曲, AUC越大 分类性能越好
    ROC 曲线图中, 越靠近 TPR 轴越好, 对图右下角的面积称为AUC  
    正常情况下 AUC 取 (0.5 - 1)    原因: 当小于0.5时 反过来预测则会大于0.5 
    TPR = TP / (TP + FN)   真正例率
    FPR = FP / (TN + FP)   反正利率
"""

# AUC
"""
AUC 被定义为ROC曲线下的面积, 取值范围在0.5 - 1 之间. 优点: AUC 作为数值可以直观评价分类器好坏, 值越大越好
    AUC = 1 是完美分类器
    0.5 < AUC < 1  优于随机猜测, 妥善设定阈值有预测价值
    AUC = 0.5 和随机猜测一样
    AUC < 0.5 比随机预测还差, 需要进行反预测   
"""

# 模型评估
"""
回归结果度量: 
    1. explained_variance_score: 可解释方差的回归评分函数
        explained_variance(yi, y(^)i) = 1 - var(y - y(^)) / var(y)
        train_pre = lin_reg.predict(X_train)

    2. mean_absolute_error: 平均绝度误差
        MAE = 1/m * |yi - y(^)i|
        mae = metrics.mean_absolute_error(true, predicted)

    3. mean_squared_error: 均方误差
        MSE = 1/m * ∑ (yi - y(^)i)**2 
        mse = metrics.mean_squared_error(true, predicted) 

    4. R方(R**2)_score: R**2 值
        R**2 = 1 - RSS / TSS = 1 - (∑(yi - y(^)i)**2 / (∑(yi - y(-))**2    y(-) 是 y 的平均值
        r2_square = metrics.r2_score(true, predicted)

精准率(ACC:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)    这里TP为预测事情发生为 对 的概率, TN为预测事情不发生 为 是 的概率 
混淆矩阵
"""

# KNN 算法
"""
KNN 算法原理
K邻近 (K-nerst neighbors, KNN)  有k个邻居,  每个样本都可以用他k个邻居来代表. 
KNN 在做分类和回归的主要区别在于最后做预测的时候的决策方式不同. 
    分类预测: 多数表决法
    回归预测: 平均值法

1. 从训练集中获取 k 个 离待测样本距离最近的样本数据
2. 根据获取得到的 k 个 样本数据来预测当前待预测样本的目标属性值
3. 默认k 等于5

KNN 算法制作步骤
1. 将训练集中的所有样例画入坐标系, 也将待测样例画入 选k值
2. 计算待测分类的内容与所有已知分类的内容欧氏距离   算距离
3. 将这些内容 排序!!! 排序!!! 取前k 个内容, 假设 k = 3, 则取前三个内容

KNN 的三要素
1. K值的选择: 一般根据样本分布选一个较小的值, 通过交叉验证选择一个比较合适的最终值, k 选小时, 误差变小, 容易过拟合, k 选大时, 训练误差变大, 导致欠拟合.
2. 距离的度量: 一般用欧氏距离
3. 决策规则: 分类模型中, 主要使用 多数表决法或者加权多数表决法; 在回归模型中, 主要使用平均值法或者加权平均法.

KNN 预测分类的规则
1. 多数表决法: 每个邻近样本的权重一样, 最后预测结果为出现最多类别的那个类
2. 加权多数表决法: 每个邻近样本的权重不一样, 一般情况下采用权重和距离成反比的方式来计算, 最终预测结果是出现权重最大的那个类别.

KNN 预测回归的规则
1. 平均值法      
2. 加权平均值法 
根据距离分之一 加起来 如果不为1之后再进行加权化, 得到权重
"""

# 编程实现KNN
import numpy as np
from torch import sigmoid

# 数据
T = [
    [3, 104, -1],
    [2, 100, -1],
    [1, 81, -1],
    [101, 10, 1],
    [99, 5, 1],
    [98, 2, 1]]

# 初始化 k 邻数据
k = 3
# 初始化待测样本
x = [18, 90]   # 这个值暂时自己给定 原本是从原数据中拿出来的
# 初始化存储列表 [[第一条样本距离, 第一条样本标签], [第二距, 第二标]]
listdistance = []
# 循环选出的 k 个数据点, 把计算结果放入储存列表
for i in T:
    dis = np.sum((np.array(i[: -1]) - np.array(x)) ** 2) ** 0.5     # 计算欧氏距离
    listdistance.append([dis, np.array(i[: -1]) - np.array(x)])

# 排序
listdistance.sort()

# 计算权重 为了防止分母为0 因此在后面加了0.001
weight = [1 / (i[0] + 0.001) for i in listdistance[: k]]       # 计算权重, 距离的倒数   1 / i[0]  取前k 个邻居    这里的k 为 k 邻近       

# # # 权重归一化, 计算出来的权重比权重和
# weight /= sum(weight)     在下面会直接归一化, 这一步可以省去

# 进行多数投票   1 / (i[0] + 0.001) for i in listdistance[: k] / sum(weight)  每个样本做归一化后的权重
pre = -1 if sum([1 / (i[0] + 0.001) / sum(weight) * i[1] for i in listdistance[:k]]) < 0 else 1      # i[1] 是标签,  sum(weight) 是对 计算权重
print(pre)      # 判断类别


# KNN算法实现的方法
"""
KNN算法的重点在于找出k 个最邻近的点, 主要方式有以下几种:
    1. 蛮力实现: 计算预测样本所有训练集的距离, 选择k个临近点; 缺点: 数据量大的时候执行效率低
    2. KD树(kd_tree): KD树算法中, 首先对训练数据进行建模, 构建KD树, 然后根据好的模型选择邻近样本数据
    3. 还有从kd_tree修改后求邻近点的算法: Ball Tree, BBF Tree, MVP Tree等
"""

# KD Tree
"""
KD Tree 是用于计算最近邻的快速便捷方法, 样本量大的时候推荐使用 加速检索效率
1. 采用从m 个样本的 n 维特征中, 分别计算n个特征取值的方差, 
2. 用方差最大的第 k 维特征 nk 作为根节点. 对小于nk 的中位数 nkv 的数划分到左子树, 大于的 nkv 划到右子树
3. 对左右子树采用同样方法找方差最大的作为根节点, 递归产生kd_tree
"""

# 构造kd_tree
"""
1. 选择x轴, 取中位数, 将坐标空间分成两个矩形
2. 对于划分后再划分, 分成多个节点
3. 分别处理所有待处理的节点
"""
# KNN 架构思路代码
import numpy as np
import pandas as pd
 
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score
 
 
class KNN():
    '''
    KNN的步骤：
    1、从训练集合中获取K个离待预测样本距离最近的样本数据；
    2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
    '''
    
    def __init__(self, k, with_kd_tree=True):
        self.k = k
        self.with_kd_tree = with_kd_tree
 
    def fit(self, x, y):
        '''
        fit 训练模型 保存训练数据
        如果with_kd_tree=True 则训练构建kd_tree
        :param x:
        :param y:
        :return:
        '''
        ###将数据转化为numpy数组的形式
        x = np.asarray(x)
        y = np.asarray(y)
        self.train_x = x
        self.train_y = y
        if self.with_kd_tree:
            self.kd_tree = KDTree(x, leaf_size=10, metric='minkowski')
 
    def fetch_k_neighbors(self, x):
        '''
        ## 1、从训练集合中获取K个离待预测样本距离最近的样本数据；
        ## 2、根据获取得到的K个样本数据来预测当前待预测样本的目标属性值
        :param x: 当前样本的特征属性x(一条样本)
        :return:
        '''
        if self.with_kd_tree:
            ## kd_tree.query([x],k=self.k,return_distance=True))
            # 返回对应最近的k个样本的下标，如果return_distance=True同时也返回距离
            # print(self.kd_tree.query([x],k=self.k,return_distance=True)[0])
 
            # 获取对应最近k个样本的标签
            index = self.kd_tree.query([x], k=self.k, return_distance=False)[0]    # 筛选函数
            # print(index)
            k_neighbors_label = []
            for i in index:
                k_neighbors_label.append(self.train_y[i])
            # print(k_neighbors_label)
            return k_neighbors_label
        else:
            ## 定义一个列表用来存储每个样本的距离以及对应的标签
            # [[距离1,标签1],[距离2,标签2],[距离3,标签3]....]
            listDistance = []
            for index, i in enumerate(self.train_x):
                dis = np.sum((np.array(i) - np.array(x)) ** 2) ** 0.5
                listDistance.append([dis, self.train_y[index]])
            # print(listDistance)
 
            ## 按照dis对listDistance进行排序
            # listDistance.sort()
            # print(listDistance)
            sort_listDistance = np.sort(listDistance, axis=0)
            # print(sort_listDistance)
            # print(type(sort_listDistance))
 
            ## 获取取前K个最近距离的样本的标签
            k_neighbors_label = sort_listDistance[:self.k, -1]
            # print(k_neighbors_label)
            ## 也可以获取前k个最近邻居的距离
            k_neighbors_dis = sort_listDistance[:self.k, :-1]
            return k_neighbors_label
 
    def predict(self, X):
        '''
        模型预测
        :param X: 待预测样本的特征矩阵（多个样本）
        :return: 预测结果
        '''
        ### 将数据转化为numpy数组的格式
        X = np.asarray(X)
 
        ## 定义一个列表接收每个样本的预测结果
        result = []
        for x in X:
            k_neighbors_label = self.fetch_k_neighbors(x)
            ### 统计每个类别出现的次数
            y_count = pd.Series(k_neighbors_label).value_counts()
            # print(y_count)
            ### 产生结果
            y_ = y_count.idxmax()
            # y_ = y_count.argmax() ##idxmax() 和 argmax 功能一样，获取最大值对应的下标索引
            result.append(int(y_))
        return result
 
    def socre(self, x, y):
        '''
        模型预测得分：我们使用准确率 accuracy_score
        :param x:
        :param y:
        :return:
        '''
        y_true = y
        y_pred = self.predict(x)
        return accuracy_score(y_true, y_pred, average='micro')   # 这里用召回和f1 需要average
 
if __name__ == '__main__':
    T = np.array([
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1]])
    X_train = T[:, :-1]
    Y_train = T[:, -1]
    x_test = [[18, 90], [50, 10]]
    knn = KNN(k=5, with_kd_tree=True)
    knn.fit(x=X_train, y=Y_train)
    print(knn.socre(x=X_train, y=Y_train))
    # knn.fetch_k_neighbors(x_test[0])
    print('预测结果：{}'.format(knn.predict(x_test)))
    print('-----------下面测试一下鸢尾花数据-----------')
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
 
    X, Y = load_iris(return_X_y=True)
    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(x_train.shape, y_train.shape)
    knn01 = KNN(k=3, with_kd_tree=True)
    knn01.fit(x_train, y_train)
    print(knn01.socre(x_train, y_train))
    print(knn01.socre(x_test, y_test))


# sklearn
from sklearn.datasets import fetch_openml    # 导入数据源
X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)     # mnist_784 由28 像素的宽和 28 像素的高 构成minist 图像
print(X.shape)    # (70000, 784)
print(y.shape)    # (70000,)
X.values[0]

# 线性回归 Liner
"""
回归算法是一种有监督学习算法
最终要求是计算出 θ 的值, 并选择最优的 θ 值构成算法
最小二乘: 
    使所有样本的预测值和真实值之间的差值最小化
    J(θ) = 1/2 * (∑ε^(i))**2 = 1/2 * (∑hθ(x^(i)) - y^(i))**2  预测值和真实之间的差值平方
误差: ε^(i) (1 <= i <= n) 服从均值为0, 方差为δ**2 的正态分布(高斯分布)
为什么多次测量取平均值越接近真实值:  因为误差服从均值为0 方差为定值的正态分布
似然函数:
    联合概率分布函数
    极大似然估计: 求参数 θ 
    目的: 达到误差最小值 
最小二乘的参数最优解(矩阵):
    参数解析式: θ = (X.T * X)**(-1) * X.T * Y
               θ = (X.T * X + λI)**(-1) * X.T * Y   # 加入超参数的最小二乘

线性回归过拟合:
    训练集和测试集的预测结果完全重合, 可能导致未来预测不准确.

为了防止过拟合, 也就是θ 值样本空间中不能过大, 可以在目标函数上加一个平方和损失:
    损失函数: J(θ) = 1/2 * (∑h(θ)(x^(i) - y^(i))**2) + λ∑θ^2 (j)

正则项(norm)/惩罚项(防止过拟合):
    λ∑θ^2 (j) 为L2正则项
    λ∑|θ|(j)  为L1正则项
    λ是超参数, 需要手动输入

Lasso 回归:
    使用L1 正则的线性回归模型称为lasso 回归
    J(θ) = 1/2 * (∑h(θ)(x^(i) - y^(i))**2) + λ∑|θ|(j)
    lasso_reg = linear_model.Lasso(alpha = 0.1)
    lasso_reg.fit(X_train, y_train)
    lasso_reg.coef_    # 系数矩阵
    lasso_reg.intercept    # 截距
    test_pre = lasso_reg.predict(X_test)
    train_pre = lasso_reg.predict(X_train)

岭回归:
    使用L2 正则的线性回归模型称为岭回归
    J(θ) = 1/2 * (∑h(θ)(x^(i) - y^(i))**2) + λ∑θ^2 (j)
    from sklearn import liner_model
    reg = linear_model.Ridge(alpha = 0.5)       # 岭回归, 超参数设置0.5
    reg.fit(X_train, y_train)                   # 训练内容
    reg.coef_                                   # 各项系数
    test_pre = reg.predict(X_test)              # 测试集预测
    train_pre = reg.predict(X_train)            # 训练集预测
    
Elasitc Net(弹性网络):
    同时使用L1 和 L2 正则的线性回归模型:
    J(θ) = 1/2 * (∑h(θ)(x^(i) - y^(i))**2) + λ * ((1-p) * ∑θ^2 (j) + (p * ∑|θ|(j))      这里的p 是概率4
    e_net = linear_model.ElasticNet(alpha = 0.5)
    e_net.fit(X_train, y_train)
    e_net.coef_
    test_pre = e_net.predict(X_test)
    train_pre = e_net.predict(X_train)

调参(超参数):
    1. 交叉验证: 将训练数据分成多份, 其中一份进行数据验证并获取最优超参数: λ 和 ρ : 比如十折交叉验证 或 五折交叉验证等
    2. 目的: 找到最优的一组参数
调参选择:
    1. 学习率的选择, 不能过大也不能过小
    2. 算法初始化参数值选择: 梯度下降法, 选择多次不同的初始值运算算法, 并返回最小损失函数值
    3. 标准化, 减少特征取值的影响
"""

# 梯度下降(BGD)
"""
场景: 用于求解无约束情况下凸函数的极小值, 是迭代类型的算法
    J(θ) = 1/(2*m) * (∑h(θ)(x^(i) - y^(i))**2)
    θ^(*) = argminJ(θ)
条件:
    1. J(θ), 凸函数, 无约束
    2. 初始化函数
    3. 沿梯度迭代, 更新θ
        θ = θ - α * ((J(θ) - J(0)) / (θ - 0))       α 为学习率, 步长
运行方式:
    在函数两侧反复横条, 越降越低, 直到迭代到最小的点


随机梯度下降算法(SGD):
目的:
    1. 优化目标函数J(θ)
    2. 定义一个初始化θ 向量
    3. 求梯度, 根据梯度更新参数, 一直迭代达到迭代停止条件
        达到一定次数停止, 差值到一定范围内
        θ(new) = θ(old) +  α * (y^(i) - h(θ)(x^(i))*x^(i)(j))
    缺点: 
        由于梯度的计算仅依赖于一个样本, 计算结果包含较大的噪声.

小批量梯度下降法(MBGD):
保证运行快, 保证最终参数训练准确率
运行方式:
    拿 b个样本(b 一般为10)的平均梯度作为更新方向

BGD SGD MBGD 的区别(在m个样本中):
    1. 每次迭代BGD 中对于参数值更新一次, SGD更新m 次, MBGD更新 m/n 次, 相当于SGD算法更新速度最快
    2. SGD每次都需要更新新的参数值, 当样本值不正常的时候, 可能会对更新产生反影响, 是在收敛处波动的.
    3. SGD中每个样本都需要更新一次参数值, 所以SGD特别适合样本数据量大的情况以及在线机器学习

代码(BGD):  
    def j(theta):       # 损失函数
        return (theta - 2)**2 - 1
    def dj(theta):      # 损失函数的导数
        return 2 * (theta - 2)
    theta = -0.75       # 初始化权重的初始点
    theta_history = [theta]
    eta = 0.01    # 步长/学习率
    epsilon = 1e-8   # 精度问题或者 eta 的设置无法令导数为0
    while True:
        gradient = dj(theta)    # 求导
        last_theta = theta      # 记录上一个theta的值
        theta = theta - eta * gradient   # 得到一个新theta
        theta_history.append(theta)
        if (abs(j(theta) - j(last_theta))< epsion):
            break
    plt.plot(plot_x, j(plot_x))
    plt.plot(np.array(theta_history), j(np.array(theta_history)))
    plt.show()
根据学习率的不同, 所得到的梯度下降不同

(SGD)  需要对样本计算完累加
    def j(theta):       # 损失函数
        return (theta - 2)**2 - 1
    def dj(theta):      # 损失函数的导数
        return 2 * (theta - 2)
    theta = -0.75       # 初始化权重的初始点
    theta_history = [theta]
    eta = 0.01    # 步长/学习率
    epsilon = 1e-8   # 精度问题或者 eta 的设置无法令导数为0
    gradient = 0
    while True:
        for i in plot_x:        # 求导
            gradient += dj(i)
        last_theta = theta      # 记录上一个theta的值
        theta = theta - eta * gradient   # 得到一个新theta
        theta_history.append(theta)
        if (abs(j(theta) - j(last_theta))< epsion):
            break
    plt.plot(plot_x, j(plot_x))
    plt.plot(np.array(theta_history), j(np.array(theta_history)))
    plt.show()
"""

# 逻辑回归 logistic_softmax 
"""
logistic 回归 softmax 回归都是 用回归思想解决分类问题
logistic 是一个线性的模型
分类或者二分类模型优先考虑逻辑回归
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear','sag','saga']
penalty = ['l2']
c_values = np.logspace(-4, 4, 15)
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=mode)
grid_search.fit(X_train, y_train)

Logistic / sigmoid函数:
    对于特征X和二分类标签y, 定义下面条件概率:
        p(y = 1| x, w, b) = 1 / (1 + e**(w.T * X + b))
        p(y = 0| x, w, b) = e**(-θ.T * X) / (1 + e**(w.T * X + b))
    合并:
        p(y|x) =  p(y = 1| x, w, b)**y * [1 - p(y = 0| x, w, b)]**(1-y)

    p = hθ(x) = g(θ.T * x) = 1 / (1 + e**(-θ.T * X)
    拟合事件发生的概率p 以及对发生或者不发生的预测
    y = sigmoid(X)

logistic回归及似然函数:
    L(θ) = p(y_hat | x; θ) = ∏ p(y^(i) | x^(i); θ) = ∏ (hθ(x^(i)))**y^(i) * (1 - hθ(x))**(1 - y)
"""

# softmax回归 可以用梯度下降法求解 θ 
"""
softmax 回归是logistic 回归的一般化, 适用于k 分类问题, 针对每个类别都有一个参数向量θ 组成二维矩阵 θ nk
本质:
    将一个k 维向量压缩成另一个k 维实数向量, 每个向量都处于 (0, 1) 之间. 相当于计算概率.
    归一化相当于对概率p 做归一化. 训练出来的每个 θ 不一样.
"""

# 特征工程
"""
定义: 所有一切为了让模型变好的数据处理方式都可以认为是特征工程的范畴
操作: 需要实际情况实际尝试使用
处理内容:
    1. 异常值处理
    2. 数据不平衡处理
    3. 文本处理: 词贷法, TF-IDF
    4. 多项式扩展: 哑编码, 标准化, 归一化, 区间放缩法, PCA(主成分分析), 特征选择---
    5. 将均值, 方差, 协方差作为特征属性. 对属性进行对数转换, 质数转换
    6. 衍生出新的特征
"""

# 数据清洗
"""
获取数据的时候尽可能需要保证标签真实值, 如果不真, 可以舍弃
步骤:
    · 去重                         drop_duplicated
    · 去除/替换不合理的值           drop/replace
    · 去除/重构不合理的字段         drop/replace
    · 数据合并                     concat append merge

缺失值填充:
    from sklearn.impute import SimpleImputer     # 正常情况按列填充  python里面每一行 我们看为一列
    impl.fit_transform(X)     # 在训练数据的时候里面 mean 值填充
    impl.statistics_          # 
    1. 均值     impl = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    2. 中位数   impl = SimpleImputer(missing_values = np.nan, strategy = 'median')
    3. 众数     impl = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    4. 常数     impl = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 1)  填充为1

fit transform fit_transform 区别:
    fit: 训练  -- 保证后面可以直接加载应用 现在实例化的对象学习后生效了
    transform: 根据fit 训练后的内容进行转换
    fit_transform : 训练后转化

模型过拟合处理方法:
    1. 惩罚项/正则项
    2. 特征提取
    3. 增加数据

欠拟合处理方法:
    1. 多项式扩展
    2. 使用更复杂模型
    3. 增加数据

归一化/标准化
使用条件:
    · 特征的单位或者大小相差大, 比其他特征大出几个数量级, 容易影响目标结果
    · 无法学习到其他特征

目的: 
    去量纲, 消除量纲

归一化
计算(每次只是一个特征一列的值内):
    x_new = (x - min(x)) / (max(x) - min(x))
    x_Nd = x_new * (mx(范围上界) - mi(范围下届)) + mi   # 归一化到指定范围内
异常点较多, 对归一化的影响:
    对于找到的最大值或者最小值可能是异常值, 导致鲁棒性较差, 只适合传统精确小数据场景
代码:
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxcaler(feature_range = (0, 1))
    scaler.fit(X)       # 学习的是每一列的最大值和最小值
    scaler.transform(X) # 转换X 数据为归一化数据
    scaler.data_max_    # 原始特征最大值
    scaler.data_min_    # 原始特征最小值

标准化计算:
    x_new = (x - mean(x)) / σ(x)
    对于异常点, 量少对于均值影响不大, 适合含有噪音的大数据场景.
代码:
    from sklearn.preprocessing import StandardScaler
    ss = StandarScaler(with_mean = True, with_std = True)    # 参数为 均值和方差是否进行规范化
    ss.fit(X)    # 学习的是每一列的均值和方差
    ss.transform(X)     # 对训练集转换X
    ss.transform(x_test)    # 转换测试集转换X

标准化和归一化的区别:
    标准化是对 数据的一列特征做标准化
    归一化是以标签(行)做归一化

特征提取:
    1. 字典特征(特征离散化)
        from sklearn.feature_extraction import DictVectorizer
        transfer = DictVectorizer(sparse = False)      
        data = transfer.fit_transform(data)    # 转换出来的数组, 是就是1 不是就是0 做了个 onehot 编码, 对数值型没有做onehot 编码

    2. 文本提取 (tfidf[缺点: 根据词频提取], onehot, 词贷法, word2ved, doc2ve)
        onehot:
            from sklearn.preprocessing import OneHotEncoder
            enc = OneHotEncoder(handle_unknown = 'ignore')      # 实例化OneHot
            X = [['Male', 1], ["Female", 3], ["Female", 2]]     # 定义数据
            enc.fit(X)    # onehot 后得到一个稀疏的矩阵  是一个转换行列过后的一个索引   好处: 节省内存空间
            enc.transform(X).toarray()   # 对上面第二个数据也进行了 OneHot
            enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])   # 反编译 能找到的返回他的值, 找不到返回None
            enc.categorise_   # 数据查看
            enc.transfrom([['Female', 1], ['male', 4]]).toarray()
            enc.get_feature_names_out(['gender', 'group'])   # 给每一列添加标题

            drop_enc = OneHotEncoder(drop = "first").fit(X)  # 删除掉第一列并进行学习
            drop_enc.transform([['Female', 1], ['male', 3]]).toarray()  # 转换删掉的 有多余特征的话会报错
            如果不想报错
            drop_enc = OneHotEncoder(drop = 'if_binary', handle_unknown = 'ignore').fit(X)      # if_binary 只有当特征是两维的时候才进行drop 
            drop_enc.transform([['Female', 1], ['male', 4]]).toarray()  # 这里改为4 不报错

            pandas里面 onehot 为 pd.get_dummies(a)   只针对里面的字符数据

        jieba:
            import jieba
            jieba.cut(X, cut_all = True)   # 切割字符, 分开字符  全模式   出现的可能性全部展示
            jieba.cut(X)                   # 精确模式
            jieba.cut_for_search(X)        # 搜索模式
            jieba.lcut(X)                  # 切好后的结果返回一个列表
            jieba.del_word("内容")         # 在后续中不会认为是一个单词
            jieba.add_word("内容")         # 在后续中, 认为是一个单词
            jieba.load_userdict("文件路径")    # 导入自定义词典库
            "/".join(jieba.cut("内容", HMM = False/True))     # 关闭/开启HMM发现新词功能
            jieba.suggest_freq(("内容"), True)      # 建议怎么分开

            import jieba.analyse
            jieba.analyse.extract_tags(X, topK = 5, withweight = True)    # 获取TF-IDF最大的五个单词

    3. 图片特征提取 数组矩阵进行提取

特征降维:
    1. 特征选择
        从原有特征中找出主要特征
        方法:
            1. Filter: 过滤法
                按照发散性或者相关性对各个特征评分, 包括: 方差选择法, 相关系数法, 卡方检验, 互信息法
                    方差选择法: 
                        from sklearn.feature_selection import VarianceThreshold
                        X = np.array
                        y = np.array
                        variance = VarianceThreshold(threshold = 0.1)
                        variance.fit(X)
                        variance.transform(X)

                    相关系数法:
                        from sklearn.feature_selection import VarianceThreshold, SelectKBest
                        from sklearn.feature_selection import f_regression
                        sk1 = SelectBest(f_regression, k = 2)    # f_regression 相关系数
                        sk1.fit(X)
                        sk1.scores_
                        sk1.transform(X)

                    卡方检验(分类任务离散特征):
                        from sklearn.feature_selection import chi2
                        sk2 = SelectKBest(chi2, k = 2)      # chi2 卡方检验
                        sk2.fit(X)
                        sk2.scores_
                        sk2.transform(X)


            2. Wrapper: 包装法
                根据目标函数, 每次(递归的过程)选择或者排除若干特征, 包括: 递归特征消除法.
                    递归特征消除法: 使用一个基模型进行多轮训练, 每轮训练后, 消除若干权值系数的特征, 再基于新特征进行下一轮训练.
                        from sklearn.feature_selection import RFE
                        from sklearn.Liner_model import LogisticRegression
                        estimator = LogisticRegression()
                        selector = RFE(estimator, step = 2, n_feature_to_select = 1)        # REF("基础学习器", 每一步消除时候消除多少个特征, 选择出多少个特征)
                        selector = selector.fit(X, y)
                        selector.support_
                        selector.n_features_
                        selector.ranking_
                        selecotr.transform(X)

            3. Embedded: 嵌入法
                先使用一些机器学习算法和模型进行训练(只训练一次), 得到各个特征的权重系数, 按系数从大到小选择特征, 包括: 惩罚项的特征选择法(L1, L2) 
                    嵌入法: 基于惩罚项的特征选择法
                        from sklearn.Liner_model import LogisticRegression
                        from sklearn.feature_selection import SelectFromModel
                        estimator = LogisticRegression(penalty = 'L2', C = 0.1, solver = 'Liblinear')    # penalty 惩罚项(L1, L2, F1)
                        sfm = SelectFromModel(estimator, threshold = 0.07)      # threshold 阈值 小于这个数的可以不要
                        sfm.fit(X, y)
                        sfm.transfrom(X)
                        sfm.estimator_.coef_        # 系数

    2. PCA 主成分分析无监督 没有y / LCA 线性判别分析法有监督 有y
        PCA 根据特征矩阵X 找期望所在投影的维度上数据方差最大, 选择方差最大的那个方向选择坐标轴
            from skleran.decomposition import PCA
            pca = PCA(n_components = 0.9, whiten = True)       # n_composition 降维后维度  whiten 标准化
            pca.fit(X)
            pca.mean_
            pca.components_
            pca.transfrom(X)

        LDA 根据分类类别投影, 投影上, 类内方差小, 类间方差大 
            from sklearn.discriminant_analysis import LinerDiscriminantAnalysis
            clf = LinerDiscriminantAnalysis(n_components = n_class - 1)    # n_claass 为 y 里面有几个不同类型的标签
            clf.fit(X, y)
            clf.transfrom(X)

        相同点:
            · 均可进行降维
            · 降维时使用矩阵分解思想
            · 符合高斯分布
        不同点:
            · LDA 有监督学习, PCA 无监督学习
            · LDA 最多降维到类别数目k-1 维, PCA没有限制
            · LDA 除了降维还可以做分类
            · LDA 选择是分类性能最好的投影, PCA 是样本点投影具有最大方法的方向

目的:
   1. 减少特征属性的个数
   2. 确保属性之间相互独立
   3. 解决多重共线性导致的空间不稳定, 泛化能力较弱
   4. 高维空间样本稀疏性, 导致模型比较难找到数据特征
   5. 防止只考虑单个变量对目标属性的影响, 从而没找出潜在关系   

kaggle 里面的内容
1. pd.drop('内容', axis = [1 or 0], inplace = True / False)    # 删除某行或者某列
2. pd.dropna()        # 删除空值
3. pd.fillna()        # 填充空值
4. pd.fillna(method = "bfill", axis = 0).fillna(0)     # 把下一行的值填充为上面的, 把剩下的填充为0
5. np.product(pd, shape)   # 行和列相乘
6. Scaling 压缩数据到 0-1 之间   标准化/归一化
    minmax_scaling(original_data, columns = [0])    # 最大, 最小的scale 数据
7. Normal distribution  正态分布标准化
    from scipy import stats
    normalized_data = stats.boxcox(pd_data)
8. pd.to_datetime(数据, format = '%m/%d/%Y')    # 数据改成年月日
"""

# k_means
"""
K-Means算法是无监督的聚类算法
做法:
    对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇
"""

# 朴素贝叶斯
"""
朴素贝叶斯一般出来文本分类问题, 统计不同词类别出现的概率, 根据这些结果判断文本属于哪个类别的概率
先验概率: 先前所知道的概率(迟到的概率)
条件概率: 根据当时条件所得的概率
后验概率: 在已经发生的情况下, 因为条件发生的概率

逻辑:
    获取数据(训练样本) --> 对每个类别计算 p(yi) --> 对每个特征属性计算所有划分条件概率  --> 对每个类别计算p(x|yi)p(yi) --> 以 对每个类别计算p(x|yi)p(yi) 作为所属类别

exp: 
p(不帅、性格不好、身高矮、不上进|嫁) = p(不帅|嫁) * p(性格不好|嫁) * p(身高矮|嫁) * p(不上进|嫁)
朴素贝叶斯算法是假设各个特征之间相互独立

平滑处理法: 平滑处理⽅法，叫作: add-one smoothing
目的: 避免概率值为0

TF-IDF
TF-IDF代表的是词频-逆文档频率，是两个度量的组合, tfidf = tf * idf
使用 TF-IDF 度量还需要做一次归一化, 一般用L2范数进行矩阵归一化
idf 逆向文件频率

高斯朴素贝叶斯:
    特征值连续时, 服从高斯分布, 计算p(x|y) 可以直接使用高斯分布的概率公式
    g(x, η, σ) = 1 / (2*π)**(1/2) * σ * exp(-(x - η)**2 / 2 * σ**2)

伯努利朴素贝叶斯:
    特征值连续, 服从伯努利分布:
    p(xk|y) = p(1|y)*xk + (1-p(1|y))(1-xk)

多项式朴素贝叶斯:
    特征服从离散的多项分布:
    p(yk) = (N(yk) + α) / (N + k*α)   ==> p(xi|yk) = (N(yk,xi) + α) / (ni + k*α)    
    加入 α 避免了模型参数为0的情况
    α = 1 时成为利普西斯(Laplace)平滑
    0<α<1 时成为Lidstone平滑
    α = 0 时不做平滑 

贝叶斯网络:
    把研究的随机变量根据是否独立, 分类绘制在一个有向图中, 形成贝叶斯网络
    定义: 有向无环图模型
    目的: 考察随机变量及其N组条件概率分布的性质
"""

# 建立 tf-idf 词频权重矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(norm = '2')    # 使用2范数归一化
tf_train_data = tfidf.fit_transform(train_content)   # 得到tfidf 值 已经训练好了
tf_train_data = tf_train_data.toarray()   #   VSM 矩阵

tf_test_data = tfidf.tranform(test_content)         # 直接使用训练好的进行 转变
tf_test_data = tf_test_data.toarray()             #   VSM 矩阵

# 朴素贝叶斯, 导入 + 代码使用
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(tf_train, train_label)
print(nb_model.score(tf_valid_data, valid_label))
# 数据导入
X_train = []     # 训练数据特征
y_train = []     # 训练数据label
X_test = []      # 测数据的特征
y_test = []      # 测试数据的label
print(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))

# 决策树
"""
决策树是一种基本的分类(逻辑回归, 朴素贝叶斯(文本分类))与回归(线性回归)方法. 
决策树基于各种情况的概率上, 通过构建树来进行分析的一种方式, 是一种直观应用概率分析的一种图解法
预测模型:  对象属性与对象值之间的映射关系
决策树是一种有监督学习的分类算法
决策树的深度是除了第一行外的其他行数的个数. 

决策树类型:
    1. 分类树  -- 分类值标签
    2. 回归树  -- 预测连续值
    3. 常用的算法: 
        · ID3:
            内容使用信息熵以及信息增益来构建, 每次迭代选择信息增益最大的特征属性作为分割属性
            特点: 1.只支持离散, 不支持连续  2.构建的是多叉树  3. 单变量决策树  4. 抗噪性差  5. 速度快实现简单  6. 不是递增  7. 只是小规模
            H(D) = -∑P(i) * log2(P(i))
            Gain = Δ = H(D) - H(D|A)

        · C4.5:
            在ID3基础上, 使用信息增益率来替代信息增益, 树的构造过程会进行剪枝优化, 能够自动完成对连续属性的离散化处理
            特点: 1.构建多分支决策树  2.可连续可离散  3.准确率高  4.实现简单  5.效率低  6.只是小规模  7.支持剪枝
            H(D) = -∑P(i) * log2(P(i))
            Gain(A) = Δ = H(D) - H(D|A)
            Gain_ratio(A) = Gain(A) / H(A)    # 这里 H(A) 为 H(X) = H(P11, P12)   在同一层里 H(Y) 都一样, 所以用H(A) 这样不仅考虑分裂后条件熵, 还考虑X的信息熵
 
        · CART:
            使用基尼系数(分类树)作为数据纯度的量化指标来构建的决策树算法, 使用GINI 增益率, 选择最大的作为分割属性
            特点: 1.分类和回归两种问题  2.CART构建是二叉树  3.支持特征属性多次使用
            Gini = 1 - ∑ P(i) ** 2
            Gain = Δ = Gini(D) - Gini(D|A)
            Gain_ratio(A) = Gain(A) / Gini(A) 
             
特征类型:
    1, 是离散值, 不要求生成二叉树, 一个属性一个分支
    2. 是离散值, 要求生成二叉树, 划分的子集进行测试, 按照属于此子集和不属于此子集分成两个分支
    3. 是连续值, 确定一个分割点split_point, 分成两个分支(左连续)
    4. 只考虑当前数据特征情况下的最好分割方式, 不能进行回溯操作

构成条件(进行属性选择度量):
    1. 树的形状
        根节点 -- 内部节点 -- 叶子节点(预测结果)
    2. 决策的阈值 θi
    3. 每个叶子的节点

树构建条件:
    1. 特征选择  选择一个根节点(root)
        · 信息增益/ 信息增益比
            集合信息的度量方式称为香农熵或者熵
            熵越大, 数据不确定性越高
            熵越小, 数据不确定性越低
            为什么熵在图像中部的时候信息熵最大?
                答: 在中间时候, 达到最大分界点
                H(X) = -∑ p(xi) * log2p(xi)
        
        · 决策树量化纯度(值越大越不纯):
            1. Gini系数
                Gini = 1 - ∑P(i)**2
            2. 熵
                H(X) = -∑P(i) * log2(P(i))
            3. 错误率
                Error = 1 - max(∑P(i))

        · 信息增益度
            规律: 信息增益度越大, 特征属性上损失的纯度越大, 该属性就在决策树上层
            计算公式: 
                Gain = Δ = H(D)[划分前的信息熵] - H(D|A)[该条件下的信息熵]    # D为目标属性, A为一个待划分的特征属性, Gain 为 特征A 对训练数据集D 的信息增益  
                Gain = Gini - 分割后的Gini(Gini左*p + Gini右*p)

        · 信息熵计算:
            g(D, A1) = h(D) - [P1 * H(D1) + P2 * H(D2) + P3 * H(D3)]   p 是该内容出现的概率  H(D) 为经验熵 
            H(D) = -∑p(xi) * log2p(xi)
            每一层叶子都需要做一次
 
    2. 树的生成
        (1). 所有特征看成一个个节点
        (2). 找到最好的分割点, 计算划分好所有子节点的 纯度信息

    3. 树的修剪
        (3). 遍历(2)得出的所有特征, 选出最优特征和最优划分方式, 得到最终子节点
        (4). 对子节点进行(2), (3)步直到叶子足够纯3

    4. 停止条件:
        1. 纯度最纯, 每个叶子只有一种类别的数据(训练集达到100%准确)     --- 可能导致过拟合
        2. 当前节点样本数小于某个阈值, 同时迭代次数打到给定值, 停止. 此处使用max(P(i)) 作为节点的对应类型

    5. 模型评估:
        1. 混淆矩阵, 准确率, 精准率, 召回率
        2. 叶子节点不纯值
            loss = ∑ |Dt| / |D| * H(t)  t∈(1, leaf)
比特化(Bits):
    假如有 A B C D 四个字符, 用 0 1 来表示四个字符  A: 00  B: 01  C: 10  D: 11
    如果有一组由 0 1 组成的数字, 他的数字不能重复使用, 只能按着顺序一个一个看 P(A) = P(B) = P(C) = P(D) = 1/4

    假设: P(A) = 1/2, P(B) = 1/4, P(C) = P(D) = 1/8  →  A: 0  B: 10  C: 110  D: 111
    E = 1*(1/2) + 2*(1/4) + 3*(1/8) + 3*(1/8)
    E = -(1/2)*log2(1/2) - (1/4)*log2(1/4) - (1/8)*log2(1/8) - (1/8)*log2(1/8)     # 算出来的结果是几个bits 位表示一个字符

    推导式:
    E = ∑-P(Ai) * log2(p(Ai))   

信息熵:
    信息熵是用来描述系统信息量的不确定度.
    规律:
        一个系统越有序, 信息熵越低, 越混乱, 信息熵越高. (有序无序: 确定性概率大小, 有序概率越高)
        高信息熵: 概率均匀 X是均匀分布等概率出现
        低信息熵: 概率波动大 有的高有的低, 不等概率出现
        越混乱, 信息量大, 越有序, 信息量小

    H(X) = -∑P(Ai) * log2(p(Ai))

条件熵 H(Y|X):
    给定条件X 的情况下, 随机变量Y 的信息熵的平均值就叫做条件熵.
    公式:
        H(Y|X) = ∑P(X = Vi) * H(Y|X = Vi) 
        H(Y|X) = H(X, Y) - H(X)

互信息I(X, Y), 条件熵(H(Y|X)), 联合熵(H(X, Y))
    H(Y|X) = H(X, Y) - H(X)
    H(X, Y) = H(X) + H(Y) + I(X, Y)
    H(Y|X) = H(X) - I(X, Y)

分类树和回归树的区别
    分类树: 信息熵, 信息增益, 基尼系数评价树的效果, 都是依据概率值判断, 预测值为叶子节点中概率最大的类别
    回归树: 节点为叶子节点中所有值的均值来作为预测值, 用MSE MAE(越小越好) 评价效果, 即均方差
    MSE = (1/n) *  ∑(yi - yi(^))**2            MAE = 1/n * ∑|yi - yi(^)|
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandarScater, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import auc, roc_curve, calssification_report
    label_encoder = LabelEncoder()      # 类别标签进行编码 
    Y = label_encoder.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 200)   # 划分测试集与训练集
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)       # 标准化
    X_test = scaler.transform(X_test)

    algo = DecisionTreeClassifier(criterion = 'gini', max_depth = None, min_samples_leaf = 1, min_samples_split = 2)     # 对决策树实例化  max_depth 构建决策树层数
    algo.fit(X_train, Y_train)
    algo.feature_importances_    # 各个权重系数
    calssification_report(Y_train, algo.predict(X_train))    # 训练数集上的分类报告
    calssification_report(Y_test, algo.predict(X_test))    # 测试数集上的分类报告
    algo.score(X_train, Y_train)    # 训练集上的准确率
    algo.socre(X_test, Y_test)    # 测试集上的准确率

    test1 = [最优test]
    algo.predict(test1)    # 预测函数
    algo.predict_proba(test1)    # 预测概率函数
    
    # 决策树可视化
    # 方法一
    # 命令: dot -Tpdf 文件名.dot -o 文件名.pdf   转换成pdf文件 
    from sklearn import tree
    with open('文件.dot', 'W') as writer:
        tree.export_graphviz(decision_tree = algo, out_file = writer)
    # 方法二
    import pydotplus
    from sklearn import tree
    dot_data = tree.export_graphviz(decision_tree = algo, out_file = None, 
                                    feature_names = ["A1", "B1", "C1", "D1"], # 特征名称
                                    class_names = ["a", "b", "c"],  # 类别名称
                                    filled = True, rounded = True,   # 使用棱角或直角, 
                                    special_characters = True, 
                                    node_ids = True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('./iris04.png')
    graph.write_pdf("./iris04.pdf")
"""

# 集成学习
"""
定义: 集成学习就是将若干个学习器(分类器, 回归器) 组合后产生新的学习器.

核心: 多样性, 差异性

弱分类器: 准确率好于随机猜想的分类器

作用: 成功保证弱分类器的多样性

常见的集成学习思想:
    1. bagging  -- RF  randomforest
        · 自举汇聚法: 原始数据集上通过有放回的抽样方式, 重新选出S个新数据来分别训练S个分类器的集成技术
        · 预测新样本分类/回归: 采用多数投票或者求平均值的方式来统计最终的分类/回归结果
        · 弱学器模型: Linear, Ridge, Lasso, Logistic, Softmax, ID3, C4.5, CART, SVM, KNN
        · 有放回, 允许数据重复
        · 差不多有 1/3 样本数不是子模型的训练数据

    2. boosting  -- Adboost, GBDT
        · 提升学习, 用于分类和回归
        · 每一步产生弱预测模型, 并权累加到总模型里
        · 损失函数是梯度方式, 就称为梯度提升
        · 意义: 一个事件存在弱预测模型, 那么可以通过提升技术得到一个强预测模型
        · 常见的模型有:
            Adaboost
            Gradient Boosting(GBT/GBDT/GBRT)
        · 步骤:
            原始样本M个 →(训练) 弱学习器 →(效果) 修改后的训练集 → --- → 最终强学习模型
            通过不断训练或者通过弱学习器直接得到最终的强学习器

    3. stacking  -- 模型融合
        · 训练一个模型用于组合其他模型
        · 可以使用任意一种算法组合
        · 两层: 一层是不同基学习器提取特征  二层是元学习器进行分类
        · 数据分层, 把每份数据都做一次预测
        · 对结果取各个模型预测结果取均值
        from mlxtend.regressor import StackingRegressor  stacking 回归
        from mlxtend.classifier import StackingClassifier  stacking 分类
        from mlxtend.feature_selection import ColumnSelector  手动指定有哪些特征

        构造基学习器 knn、RF、softmax、GBDT。。。。
        knn = KNeighborsClassifier(n_neighbors=7)
        softmax = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial', fit_intercept=False)
        gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
        rf = RandomForestClassifier(max_depth=5, n_estimators=150)

        构造元学习器
        lr = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial')

        构造stacking学习器
        方法1. 
            stacking02 = StackingClassifier(classifiers=[knn, softmax, gbdt, rf],
                                    meta_classifier=lr,
                                    use_probas=True,
                                    average_probas=False)
        方法2.
            stacking01 = StackingClassifier(classifiers=[knn, softmax, gbdt, rf],
                                meta_classifier=lr)
        方法3. 
            pipe_knn = Pipeline([('x', ColumnSelector([0, 1])),
                     ('knn', knn)])
            pipe_softmax = Pipeline([('x', ColumnSelector([2, 3])),
                                    ('softmax', softmax)])
            pipe_rf = Pipeline([('x', ColumnSelector([0, 3])),
                                ('rf', rf)])
            pipe_gbdt = Pipeline([('x', ColumnSelector([1, 2])),
                                ('gbdt', gbdt)])
        
            stacking03 = StackingClassifier(classifiers=[pipe_knn, pipe_softmax, pipe_rf, pipe_gbdt],
                                            meta_classifier=lr)
        
        训练与比较:
        scores_train = []
        scores_test = []
        models = []
        times = []
            for clf, modelname in zip([knn, softmax, gbdt, rf, stacking01, stacking02, stacking03],
                          ['knn', 'softmax', 'gbdt', 'rf', 'stacking01', 'stacking02', 'stacking03']):
                print('start:%s' % (modelname))
                start = time.time()
                clf.fit(x_train, y_train)
                end = time.time()
                print('耗时：{}'.format(end - start))
                score_train = clf.score(x_train, y_train)
                score_test = clf.score(x_test, y_test)
                scores_train.append(score_train)
                scores_test.append(score_test)
                models.append(modelname)
                times.append(end - start)  

        结果查看
        print('scores_train:', scores_train)
        print('scores_test', scores_test)
        print('models:', models)

stacking实现:
    1. 学习L个学习器, 假定全是逻辑回归LR模型---L个学习器可以是不同算法
    lr10: z0 = lr10(x0, x1, x2, x3, ....., xn)
    lr11: z1 = lr11(x0, x1, x2, x3, ....., xn)
    lr12: z2 = lr12(x0, x1, x2, x3, ....., xn)
    lr13: z3 = lr13(x0, x1, x2, x3, ....., xn)
    ...
    lr1L: zL = lr1L(x0, x1, x2, x3, ....., xn)
    2. 学习元模型, 假定也是逻辑回归LR模型
    lr20: y = lr20(z0, z1, z2, ....., zL)
    stacking缺点:
        -1. 学习器元模型和训练安全独立, 两个阶段没影响
        -2. 在训练时候, 实际上数据是进行划分的
    深度学习/神经网络:
        是stacking的进一步发展, 不同:
            -1. 深度学习中各个层次之间相互影响, 相互促进
            -2. 全量数据均参与整个模型训练
            -3. 深度学习中各个子学习器以及元模型全部是同结构线下回归模型+几号函数

集成学习的好处:
    1. 弱分类器有差异性, 导致边界不同, 合并后---减少错误率, 实现更好的效果
    2. 对于数据集过大或者过小, 可以进行划分和有放回的操作产生不同数据子集, 使用数据子集训练不同的分类器, 最终合并为一个大的分类器
    3. 数据比较复杂, 用线性模型很难描述清楚, 可以训练多个模型, 再进行融合
    4. 对多个异构特征时很难融合, 可以考虑每个数据集构建一个分类模型, 然后多个模型融合
    5. 解决数据不均衡, 指标签 y 不均衡
"""

# bagging 回归
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

# 数据
df = pd.DataFrame([[1, 10.56],
                   [2, 27],
                   [3, 39.1],
                   [4, 40.4],
                   [5, 58],
                   [6, 60.5],
                   [7, 79],
                   [8, 87],
                   [9, 90],
                   [10, 95]],
                  columns = ['X', 'Y'])

# 循环弱学习器
M = []   # 存储学习器
line_num = 20  # 弱学习期次数
for i in range(line_num):
    tmp = df.sample(frac=1.0, replace=True)  # 抽样
    X = tmp.iloc[:, :-1]
    Y = tmp.iloc[:, -1]
    model = DecisionTreeRegressor(max_depth = 1)
    model.fit(X, Y)
    model

    M.append(model)

###做预测
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

model1 = DecisionTreeRegressor(base_estimator = 'Linear', max_depth = 1)
model1.fit(x, y)

# y_hat_01 = model1.predict(x)
# print(y_hat_01)
# print(model1.score(x, y))
# print("-" * 100)
res = np.zeros(df.shape[0])

for j in M:
    res += j.predict(x)

y_hat = res / line_num

print(y_hat)
print('R2:', r2_score(y, y_hat))

# 随机森林
"""
RF 随机森林优点:
    1. 训练可以并行优化(决策树独立并行), 对大规模样本有优势
    2. 由于随机选择决策树划分特征列表, 在高纬度还存在比较高的训练性能
    3. 给出各个特征的重要性列表
    4. 存在随机抽样, 训练出来的模型方差小, 泛化能力强, 可以缓解过拟合
    5. 实现简单
    6. 对部分特征缺失不敏感
缺点:
    1. 噪音较大的特征上(数据特别异常), RF模型容易过拟合
    2. 取值较多的划分特征对RF的决策会产生更大影响, 会影响模型效果

在bagging 策略的基础上进行修改后的一种算法
    1. 从原始样本中Bootstrap采样选出n个样本
    2. 使用抽出来的子集训练决策树, 从中选出k个属性, 选出最佳分割属性作为前节点的划分属性, 按照这种方式迭代创建决策树
    3. 重复以上步骤m次, 建立m个决策树
    4. 这m 个决策树形成随机森林, 通过投票表决结果决定数据属于哪一类

RF 算法推广:
    分类, 回归, 特征转换, 异常点检查
    1. Extra Tree  -- 有监督 分类/回归
        · 每个子决策树采用原始数据集训练
        · 随机选择特征来划分决策树
        · 因为随机选择特征划分, 规模可能大于决策树, 方差比 RF 进一步减少, 泛化能力(鲁棒性)比RF强

    2. Totally Random Trees Embedding(TRTE)  -- 特征转换  (还有PCA, poly, LDA)
        · 非监督数据转化, 将低维数据集映射到高维, 更好应用于回归 (编码后的特征为稀疏矩阵)
        · 类似于RF + KDTree算法方法, 建立T个决策树来拟合数据, 构建完成后每个数据在T 个决策树中叶子节点位置就定下来了, 将位置信息转化为向量
        
    3. Isolation Forest
        和RF 的区别:
            · 随机采样中, 一般只需要少量数据
            · IForest 会随机选择一个划分特征, 并对划分特征随机选择一个阈值
            · IForest 算法构建的决策树一般深度max_dept 比较大
        目的: 
            检测异常点, 区分异常就行, 不需要大量数据
        方法:
            将测试集样本X 拟合到m 棵决策树上, 计算每个叶子节点的深度ht(X), 从而算出平均h(x)
                p(s, m) = 2 ** (- h(x)/c(m))
                c(m) = 2*ln(m-1) + ζ - 2*((m-1) / m)    # m 为样本数   ζ 为欧拉常数
            p(s, m)越接近1, 是异常点的概率越大, 正常样本点的概率为0
"""

# boosting
"""
Adaboost 提升算法

最终会将基分类器的线性组合成为一个强分类器:
    分类误差小的基分类器给大的权值
    分类误差大的基分类器给小的权值
    弱学习器无要求
    f(X) = ∑ αm * Gm(X)

最终分类器是在线性组合的基础上进行sign函数转换:
    G(X) = sign(f(x)) = sign [∑ αm * Gm(X)]

Adaptive Boosting 是一种迭代算法, 每轮会在训练集上产生一个新的学习器, 然后预测评估(通过权重)重要性
方法:
    每一次训练依据前面的结果对样本X的权重进行改变, 预测正确的样本权重小, 预测错误的样本权重大
权重:
    指的是当前学习器的权重, 是前面预测错误后, 进行再次训练得到的权重
    样本权重 wi: 每一次训练依据前面的结果对样本X的权重进行改变, 预测正确的样本权重小, 预测错误的样本权重大
    学习器权重 αm: 错误率越小权重越大, 错误率越大, 权重越小
规律:
    样本点被预测的越正确, 样本权重降低, 否则提高样本权重.
    权重越高的样本在下一个迭代训练中所占的权重就越大, 也就是说, 越难区分的样本在训练过程中会变得越重要
停止:
    1. 错误率足够小
    2. 达到一定迭代次数

算法原理(学习器与损失函数):
分错的样本会带来一个更大的损失: 所以目的为 减少损失
    最终强学习器:
        G(X) = sign(f(x)) = sign [∑ αm * Gm(X)]    # m为下标, 表示m 个学习器
    损失函数(以错误率作为损失函数):
        越小越好
        ∑wi = 1
        loss = ∑ wi * I(G(Xi) != yi)     # I(G(Xi) != yi) = 1 分错了给一个损失, 分对了没有损失 函数为0  
    损失函数(上界):
        loss = ∑ wi * I(G(Xi) != y) <= ∑wi * e**(-yi * f(X))
    第 k-1 轮的强学习器:
        f(k-1[下标])(X) = ∑ αj * Gj(X)    j ∈[1, k-1]
    第 k 轮的强学习器:
        fk(X) = ∑ αj * Gj(X)     fk(X) = f(k-1[下标])(X) + αk * Gk(X)     
    损失函数:
        loss(αm, Gm(X)) = ∑wi * e**(-yi(f(m-1[下标])(X) + αm * Gm(X))) =  ∑wmi^(-) * e**(-yi * αm * Gm(X))    # 样本的权重 w 学习器的权重 α  Gm(X) 是弱学习器
    损失达到最小值时的最终解:
        loss(αm, Gm(X)) = ∑wmi^(-) * e**(-yi * αm * Gm(X))
        规律: 让Gm(X) 学习的时候让误差率最小 
    为了让误差率最小, 可以认为G越好误差越小:
        G^(*)m(X) = min(∑wmi^(-) * I(yi != Gm(Xi)))
        εm = P(Gm(X) != y) = ∑wmi^(-) * I(yi != Gm(Xi)) =  ∑wmi^(-)
    对于 αm 而言, 求导后令导数为0, 可以的公式
        αm^(*) = 1/2 * ln((1 - εm) / εm)

算法子模型权重系数求解:
    loss(αm, Gm(X)) = e**(-αm) + εm * e**(αm) - εm * e**(-αm)

Adaboost算法构建过程
    1. 假设训练数据集 
        T = {(X1, Y1), (X2, Y2),----, (Xn, Yn)}
    2. 初始化训练数据权重分布:
        D1 = (w11, w12, ----, w1i, ----, w1n)    # w1i = 1 / n
    3. 使用具有权值分布Dm的训练数据集学习, 得到基本分类器
        G(x): x → {-1, 1}
    4. 计算Gm(X)在训练集上的分类误差:
        εm = P(Gm(X) != y) = ∑wmi^(-) * I(yi != Gm(Xi)) =  ∑wmi^(-)
    5. 计算Gm(X)模型的权重系数:
        αm = 1/2 * ln((1 - εm) / εm)
    6. 权重训练数据集的权值分布
        Dm+1 = (w(m+1)1, w(m+1)2, ----, w(m+1)i, ----, w(m+1)n )    # w(m+1)i = wmi / Zm * e**(-yi * αm * Gm(X))
    7. 这里的Zm是规范化因子(归一化):   这里的归一化和knn里的归一化相同 是样本权重
        Zm = ∑wmi * e**(-yi * αm * Gm(xi))
    8. 构建基本分类器线性组合:
        f(x) = ∑ αm * Gm(x)      m ∈ [1, M]    这里
    9. 得到最终分类器:
        G(X) = sign(f(x)) = sign(∑ αm * Gm(x))

Adaboost scikit-learn 相关参数
     参数                            AdaBoostClassifier                                                                             AdaBoostRegressor
base_estimator         弱分类器, 默认CART分类树DecisionTreeClassifier                                                   弱回归器, 默认CART回归树 DecisionTreeRegressor
algorithm              SAMME(样本集分类效果作为弱分类器权重) 和 SAMME.R(预测概率大小做权重必须有predict_proba方法);         不支持
lasso                  不支持                                                                                          指定误差计算, linear, square, exponentail
n_estimators           最大迭代次数, 值过小可能欠拟合, 值过大可能过拟合, 一般50- 100 合适
learning_rate          指定每个弱分类器的权重缩减系数V, 默认值为1, 一般从一个较小的值开始调参, 值越小就说明需要更多的弱分类器

f(x) = ∑ αm * Gm(x)      m ∈ [1, M]   --------(添加缩减系数v)------→    f(x) = ∑ v * αm * Gm(x)       目的: 降低过拟合, 欠拟合

优点:
    1. 可以处理离散值和连续值    2. 模型鲁棒性比较强    3. 解释强, 结构简单
缺点:
    1. 对异常值敏感, 迭代过程中获得较高的权重值, 影响模型效果
"""

# Gradient Boosting
"""
梯度提升迭代决策树GBDT

GBDT 是Boosting算法的一种, 也是迭代, 要求弱学习器必须是回归CART回归树模型, 并且训练的时候, 要求样本损失尽可能的小

通用内容: 所有GBDT 算法中 底层都是回归树, 可以做分类任务也可以做回归任务

别名: GBT, GTB, GBRT, GBDT, MART

直观理解:
    1. 不断进行残差计算进行预测 之后用残差预测 不停迭代
    2. 预测值是每个部分预测值的和y^ = ym1^ + ym2^ + ···· +  
    3. 强学习器是所有弱学习器的和
    4. 给定步长step时, 构建树的时候使用 step * 残差作为输入值, 这种方式可以防止过拟合发生
    5. 通过残差影响预测值

构成:
    1. DT(Regression Decistion Tree), GB(Gradient Boosting)和Shrinkage(衰减)
    2. 由多棵决策树构成, 所有树的结果累加起来就是最终结果
    3. 迭代决策树和随机森林的区别:
        · 随机森林抽取不同的样本构建不同的子树, 也就是说第m颗树的构建和钱 m - 1 棵树没有关系
        · 迭代决策树在构建子树时, 使用之前子树构建结果的残差构建下一棵子树, 最终构建按照子树构建的顺序进行预测, 并将预测结果累加

思路:
目标函数:
    func = ∑ L(yi, y^**(t)i)
    1. 在给定向量X 和输出 Y 组成的若干训练样本(Xi, Yi)中 找到近似函数F(X) 使损失函数L(Y, F(X))的损失值最小
    2. 损失函数一般采用最小二乘或者绝对值损失函数
        L(y, F(X)) = 1/2 * (1 - F(x))**2      L(y, F(X)) = |y - F(X)|
    3. 最优解:
        F*(X) = argmin(L(y, F(X)))
    4. 假定F(X)是一族最优基本函数fi(x)的加权和:
        F(X) = ∑ fi(x)  ----防止学习器能力过强给定一个缩放系数---->   F(X) = v * ∑ fi(x)
    5. 根据决策树算法思想扩展得到Fm(X), 求解最优f:
        Fm(X) = Fm-1(X) + argmin( ∑ L(yi, Fm-1(Xi)+Fm(Xi)))        i ∈ [1,, n]
    6. 之后进行梯度计算
    7. 给定常数F0(X), 在构建第一棵树前面加, 因为第一棵树前面没有树
        F0(X) = argmin(∑L(yi, c))       # c是样本的均值 
    8. 计算损失函数的负梯度值:
        yim = -[δ L(y, F(X)) / δF(X)] = y - Fm-1(X)     # 设置为最小二乘时, 负梯度值就是残差
    9. 拟合残差, 找到一个CART回归树, 得到第m棵树
        cmj = argmin(∑ L(yiim, c)         表示第m 棵树有多少个叶子j 是一个决策树 预测值是cj
    10. 更新模型:
        Fm(X) = Fm-1(X) + cmj * I (x ∈ leafmj)  ===>  F(X) = F0(X) + ∑(m ∈ [1, M]) ∑(j ∈ [1, |leaf|]) cmj * I (x ∈ leafmj)

GBDT回归和分类的区别:
    1. 选择不同损失函数, 不同负梯度, 不同模型初始采用
    2. 回归算法一般选择最小二乘
    3. 损失函数
        回归:
            均方误差:
                损失函数:  L(y, Fm(X)) = 1/2 * (y - Fm(X))**2
                负梯度值: yim = yi - Fm-1(X)
                初始值: 一般采用均值作为初始值
            绝对误差:
                损失函数:  L(y, Fm(X)) =|y - Fm(X)|
                负梯度值: yim = sign(yi - Fm-1(X))
                初始值: 一般采用中位数作为初始值
        分类:
            二分类:
                损失函数: L(y, Fm(X)) = -(yln(pm) + (1-y)ln(1-pm)) ---逻辑回归     # pm = 1 / 1 + e**(-Fm(X))  sigmoid
                负梯度值: yim = yi - pm
                初始值: 采用ln(正样本数 / 负样本数)作为初始值
            多分类:
                损失函数: L(y, Fm(X)) = -∑yk * ln(pk(X)) ----交叉熵     # pk(X) = exp(fk(X)) / ∑exp(fi(x))  softmax
                负梯度值: yiml = yil - pml(X)
                初始值: 一般用0为初始镇

    from sklearn.ensemble import GradientBoostingClassifier   梯度分类
    from sklearn.ensemble import GradientBoostingRegressor  梯度回归
""" 

# bagging 和 boosting 的区别
"""
1. bagging 样本有放回抽样; boosting每一轮训练集不变, 每个样例在分类器权重改变或者y属性发生改变
2. bagging 等权重; boosting根据错误率不断调整权重
3. bagging 所有预测权重相同; boosting 对于误差小的有更大权重
4. bagging 并行; boosting 串行
5. bagging 目的减小模型方差; boosting 减少模型偏度
6. bagging 每个分类模型都是强分类器, 方差过高是过拟合; boosting 每个都是弱分类器, 偏度过高是欠拟合
"""

# XGBoost
"""
import xgboost as xgb
相当于对每个叶子做一个惩罚项
目标函数:
    F = func = ∑ L(yi, y^**(t)i) + ∑ Ω(fi)
    ft(x) = wq(x)   决策树, 叶子的结果, q(x)表示第几个叶子
    Ω(fi) = r * T(对叶子的惩罚项, T是深度) + 1/2 * λ * ∑ wj**2 

t次迭代后:
    y^**(t)i = y^**(t-1)i + ft(Xi)

推出目标函数:
    loss = ∑ L(yi, y^**(t)i) + ∑ Ω(fi)
    loss = ∑ L(yi, y^**(t)i) + r * t + 1/2 * ∑ wj**2  加入  l2正则
每个叶子节点j 集合为 I:
    Ij = {i|q(xi) == j}

当树的结构确定以及对应的损失最小时, 确定树结构的方法:
    1. 穷举法: 列出所有的选出最小的
    2. 贪心法: 选择每一个分裂点, 计算操作前后增益, 选择增益最大的方式进行分裂

特征:
    1. 列采样, 借鉴随机森林, 目的: 降低过拟合减少计算量
    2. 支持对缺失值自动处理
    3. XGBoost 支持并行, 在计算特征Gain 时会并行执行, 树的构建过程中是串行
    4. 加入正则项, 使最终模型不容易过拟合
    5. XGBoost 学习器支持CART, 线性回归, 逻辑回归
    6. 支持自定义损失函数
"""
import pandas as pd
import xgboost  as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split
datas = pd.read_csv("路径", index_col='0')
"""回归数据"""
X = datas.iloc[:, :-1]
y = datas.iloc[:, -1]
"""数据划分"""
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1412)
xgb_sk = XGBRegressor(random_state=1412)
xgb_sk.fit(x_train, y_train)
xgb_sk.score(x_test, y_test)
# sklearn交叉验证三步走
xgb_sk = XGBRegressor(random_state=1412)
"""定义所需要交叉验证方式"""
cv = KFold(n_splits=5, shuffle=True, random_state=1412)
result_xgb_sk = cross_validate(xgb_sk, X, y, cv=cv
                            , scoring='neg_root_mean_squard_erroor' # 负根据均方误差
                            , return_train_score=True
                            , verbose=True
                            , n_jobs=-1)
def rmse(result, name):
    return abs(result[name].mean())
rmse(result_xgb_sk, 'train_score')
rmse(result_xgb_sk, 'test_score')
xgb_sk = XGBRegressor(max_depth=5, random_state=1412).fit(X, y)
xgb_sk.feature_importances_
"""查看建立多少棵树"""
xgb_sk.get_num_boosting_rounds()    # -->100  一颗一颗树建立加起来
xgb_sk.get_params()
"""数据转换"""
data_xgb = xgb.DMatrix(X, y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1412)
d_train = xgb.DMatrix(x_train, y_train)
d_test = xgb.DMatrix(x_train, y_train)
params = {"max_depth": 5, "seed": 1412}
reg = xgb.train(params, data_xgb, num_boost_round=100)      

# 实例
from sklearn.datasets import load_breast_cancer, load_digits

"""二分类"""
x_binary = load_breast_cancer().data
y_binary = load_breast_cancer().traget
data_binary = xgb.DMatrix(x_binary, y_binary)
"""多分类"""
x_multi = load_digits().data
y_multi = load_digits().traget
data_multi = xgb.DMatrix(x_multi, y_multi)
"""交叉验证"""
params1 = {"seed": 1412, "objective": "binary: logistic", "eval_metric": "logloss"}    #二分类交叉熵
clf_binary = xgb.train(params1, data_binary, num_boost_round=100)       # 迭代次数, 树的数量

params2 = {"seed": 1412, "objective": "binary: logistic", "eval_metric": "mlogloss", "num_class": 10}   # 多分类交叉熵
clf_multi = xgb.train(params2, data_multi, num_boost_round=100)

"""预测与评估"""
y_pred_binary = clf_binary.predict(data_binary)
y_pred_multi = clf_multi.predict(data_multi)
"""转换概率"""
y_pred_binary[:20]
y_pred_multi

"""评估-->混淆矩阵"""
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import log_loss as logloss
(y_pred_binary > 0.5).astype("int")
ACC((y_pred_binary > 0.5).astype(int))
ACC(y_multi, y_pred_multi)

logloss(y_binary, y_pred_binary)
"""交叉验证"""
param2 = {'seed': 1412,
          'objective': 'multi: softmax',
          'num_calss': 10}
result = xgb.cv(param2, data_multi, num_boost_round=100,
                metrics=('mlogloss'),   # 交叉验证评估指标由metrics控制
                nfold=5,                # 交叉验证中所需参数, nfold=5 表示5折交叉验证
                seed=1412)              # 交叉验证随机数种子, 管理boosting过程中随机数种子
result # 返回多类交叉熵损失

"""分类多参数"""
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.get_params()
clf = XGBClassifier(
        objective='multi: softmax',
        eval_metric='mlogloss',
        num_calass=10
#       ,use_label_encoder=False
)
clf = clf.fit(x_multi, y_multi)
clf.predict(x_multi)
clf.predict_proba(x_multi).shape
clf.score(x_multi, y_multi)

# SMOTE 合成采样过程
"""
基本思想:
    对少数类样本分析, 并根据少数类样本人工合成新样本添加到数据集中
    1. 对于少数类中每个样本, 得到k近邻
    2. 根据样本不平衡比例设置一个采样比例确定采样倍率
    3. 对于随机选出的近邻样本xi, 分别与原样本按照如下公式构建新样本
        x_new = x0 + uniform(0, 1) * (x0 - x1)

Border-line SMOTE
不考虑周边情况容易出现两个问题:
    1. 新合成的样本不会提供太多有用的信息
    2. 新合成的样本与周围多数样本产生大部分重叠, 分类困难

样本点选取:
    1. safe: 样本周围一半以上均为少数类样本
    2. Danger: 样本周围一半以上均为多数类样本
    3. Noise: 样本周围都是多数类样本, 视为噪音

SMOTE 和 Border-line SMOTE 的区别:
    Border-line SMOTE 只会在 Danger 状态的样本中随机选择, 只会对那些靠近边界的少数类样本进行人工合成样本
    SMOTE 不论对哪种数据都一视同仁, 不做区分
    from imblearn.over_sampling import BorderlineSMOTE
    sm = BorderlineSMOTE(random_state = 42, kind = 'borderline-1')
    X_res, y_res = sm.fit_resample(X, y)
    Counter(y_res)  # 查看数据形状
"""

# SVM
"""
SVM 支持向量机, 也称支持向量网络 强学习器
可以进行:
    有监督学习:
        线性二分类与多分类
        非线性二分类与多分类
        普通连续型变量的回归
        概率连续型变量的回归
    无监督学习:
        支持向量聚类
        异常值检测
    半监督学习:
        转导支持向量机

KKT条件:
    KKT 条件是泛拉格朗日乘子式的一种形式, 主要用在优化函数中存在不等值约束情况下的最优化求解方式
    L(x, α, β) = f(x) + ∑αi * hi(x) + ∑ βj * gj(x)     i ∈ [1, p]   j ∈ [1, q]
    然后去: max(L(x, α, β))  f(x) 是 L(x, α, β) 的上界
           ∴ min(x) max(β) L(x, α) = max(β) min(x) L(x, α)
 
感知机(感知器)模型:
    原理: 用一条线将数据分割

    前提: 数据线性可分

    解决问题: 二分类问题  ---->  通过 ovo 或者 ovr  训练多个二分类模型 ----> 达到多分类

    模型: y^ = sign(Θ * x) = 1, Θ * x > 0 ;  -1, Θ * x < 0

    分类正确: y*(Θ*x) > 0 预测值和真实值同号

    损失函数: 
        只记录错误的样本
        L = - ∑ y**(i) * Θ * x**(i) / ||Θ||2 (相当于对Θ 进行了归一化) = - ∑ y**(i) * Θ * x**(i) 

线性可分SVM(硬间隔, 要求函数距离一定是1):
    让离超平面比较近的点尽可能的远离这个超平面
    · 思路:
        1. 构造约束优化问题:
            min 1/2 * ∑ βi * βj * y^i * y^j * x^i.T * x^j - ∑ βi
            s.t : ∑ βi * y^i = 0
        2. 使用SMO算法求出上优化中最优解 β*
        3. 找出所有支持向量集合S: S = {(x^i, y^i), βi > 0, i = 1, 2, 3, -----, m}
        4. 更新参数w*, b*的值:
            w* =  ∑ (βi*) * y^i * x^i                               s.t : ∑ βi * y^i = 0
            b* = 1/s * y**s - ∑ (βi*) * y^i * x^i.T * x**s          这里的xi 是支持向量
        5. 构建最终分类器:
            f(x) = sign(w* * x + b*)
    · 线性可分:
        数据集中找出一超平面, 将两组数据分开, 这个数据集线性可分
    · 线性不可分:
        数据集中找不出一超平面将两组数据分开, 这个数据集线性不可分
    · 分割超平面:
        将数据集分割出来的直线/平面叫分割超平面 
    · 支持向量:
        离分割超平面最近的那些点叫支持向量
    · 间隔:
        支持向量到分割超平面的距离叫间隔
        |w.T*x + b| / ||w||2 = 1 / ||w||2       限定条件st: y^i * (w.T*x^i+b)>=1, 这里的x 为支持向量
    · 函数距离:
        |w.T*x + b| = c
        x 对应的值不一样, 两个值直接的距离称为函数距离
        支持向量到超平面的函数距离一般为1
        最后取距离最大化(max(1 / ||w||2 )), 条件: 所有的点分正确
        max(1 / ||w||2 ) = min(1/2 * ||w||2**2)
    · 转换:
        将此时的函数用KTT条件转换为拉格朗日函数
        L(x, b, β) = 1/2 * ||w||2**2 + ∑ βi * [1 - y**i*(W.T * x) + b]      (βi > 0)
    
SVM软间隔模型:
    每个样本引入松弛因子(ζ), 使函数距离加上松弛因子后距离大于1 
        y^i * (w.T * x^i + b) >= 1 - ζ      i ∈ [1, m], ζ >= 0
    如果松弛因子越大, 离超平面越近, 大于1, 表示分错
       min(1/2 * ||w||**2 + C * ∑ ζi)       C 是超参数
    思路:
        · 选择一个惩罚系数C>0, 构造约束优化问题: 
            min 1/2 * ∑ βi * βj * y^i * y^j * x^i.T * x^j - ∑ βi
            s.t : ∑ βi * y^i = 0
        · 使用SMO算法求出上式优化中对应最优解β
        · 找出所有支持向量集合S: S = {(x^i, y^i), 0< βi < C, i = 1, 2, 3, -----, m}        
        · 更新参数w*, b*的值:
            w* =  ∑ (βi*) * y^i * x^i
            b* = 1/s * y**s - ∑ (βi*) * y^i * x^i.T * x**s          这里的xi 是支持向量
        · 构建最终分类器:
            f(x) = sign(w* * x + b*)
    处理问题:
        1. 可处理有异常点的分类模型构建问题
        2. 通过加入松弛因子增强泛化能力
        3. 给定惩罚项C(超参数), 提高准确率

非线性可分SVM:
    定义一个从低维特征空间到高纬度特征空间的一个映射: 
        min 1/2 * ∑ βi * βj * y^i * y^j * x^i.T * x^j * ψ(x^i) * ψ(x^j) - ∑ βi         核函数: k = ψ(x^i) * ψ(x^j)

总结:
    核心思想: 离超平面近的点离超平面距离越大, 靠近的点叫支持向量机
    解决非线性可分: 加入核函数
    推导过程:
        l(β) = 1/2 * ||w||2**2 + ∑ βi * [1 - y**i*(W.T * x) + b]

hard margin 硬间隔分类
使用条件:
    1. 仅在线性可分的情况下使用
    2. 对异常点非常敏感, 有异常值, 泛化能力不会很好

sotf margin 软间隔分类
寻找一个数据间良好的平衡, 通过C 参数控制平衡, 较小的c 值街道更宽, 但是间隔侵犯会更多
如果SVM 出现了过拟合, 则可以尝试通过降低c 的值来进行正则化
代码:
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandarScaler

    scaler = StandarScaler()
    scaler.fit(X)
    x_standard = scaler.transform(X)

    svc = LinerSVC(c = 1e9)
    svc.fit(x_standard, y)

    def plot_decision_boundary(model, axis):
        x0, x1 = np.meshgrid(np.linspace(axis[0], axis[1], int((axis[1] - axis[0])*100)).reshape(1, -1),
                             np.linspace(axis[2], axis[3], int((axis[3] - axis[2])*100)).reshape(1, -1))
        x_new = np.c_[x0.ravle(), x1.ravel()]
        y_predict = model.predict(x_new)
        z = y_predict.reshape(x0.shape)

        plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    plot_decision_boundary(svc, axis=[-3, 3, -3, 3])    # 画出图像

SVC
代码:
    from sklearn.svm import SVC
    from skleran.pipeline import Pipeline
    import numpy as np
    from sklearn.preprocessing improt StandarScaler
    from sklearn.metrics import mean_squared_error

    # Pipline 可替换为
    scaler = StandarScaler()
    scaler.fit_transform(X)
    svc = SVC(kernel = 'poly', degree = 3, foef0 = 1, C =5)
    svc.fit(scaler_X, y)
    svc.predict(X)

    poly_kernel_svm_clf = Pipline([
        ('scaler', StandarScaler()),
        ('svm_clf', SVC(kernel = 'poly', degree = 3, foef0 = 1, C =5))    # kernel 核函数
    ])
    poly_kernel_svm_clf.fit(X, y)
    y_predict = poly_kernel_svm_clf.predict(X)
    mean_squared_error(y, y_predict)

SMO(序列最小优化算法)   
解决问题: SVM 训练过程中所产生的优化问题算法
公式: 
    min(1/2 * ∑ βi * βj * y^i * y^j * x^i.T * x^j * k(x^i, x^j) - ∑ βi)
    st(限定条件): ∑ βi * y^i = 0
    得到最终分割超平面:
        g(x) = wx + b =  ∑ βi * y^i * k(x^i, x) + b 
                   | > 1 , {x^i, y^i| βi = 0}
    y^i * G(x^i) = | = 0 , 0 < βi < C               
                   | < 1 , {x^i, y^i| βi = C}
思路:
    初始化一个β, 满足对偶问题(KKT和拉格朗日乘子)的两个初始限制条件;
    不断优化β , 使得分割超平面满足g(x) 条件, 优化过程中始终保证β 满足两个初始限制条件
        用老的β 进行计算新的β
    目的: 让g(x)满足目标条件
                 | H, β2_new_unt > H
        β2_new = | β2_new_unt, L < β2_new_unt < H
                 | L, β2_new_unt < L
        β1_new = β1_old + y^1 * y^2 *(β2_old - β2_new)

SVR:
    目的:
        为了尽量拟合一个线性模型 wx + b , 从而可以定义常量 eps(误差值) > 0
        |y - wx - b| <= eps 表示没有损失
    公式:
        min(1/2 * ||w||2**2)
        st: |y^i - w.T * x^i - b| <= ε     i = 1, 2, 3, ....., m
    加入松弛因子 ζ>0, 目标函数和限定条件为:
        min(1/2 * ||w||2**2) + C * ∑(ζiv + ζi^)
        st: - ε - ζiv <= y^i - w.T * x^i - b <= ε + ζi^

核函数
作用:
    使原本线性不可分数据变得线性可分, 把数据从原始空间映射到一个更高维空间, 使得样本在这个特征空间内线性可分.
    对任意的两个低纬度向量x, z, 有  k(x, z) = ψ(x) * ψ(z)  称函数 k(x, z) 为核函数
    相当于低维空间上的计算量等价于特征在做维度扩展后点乘的结果
    通过核函数可转化为一个对称阵
    主要作用: 让计算的结果和高维计算结果保持一致
    好处: 减少计算量

方法:
    1. non-liner polynomialFeatures
        from sklearn.preprocessing import  PolynomialFeatures
        poly = PolynomialFeatures(degree = 3)    # degree 设置维数
        poly.fit(x)
        x2 = poly.transfrom(x)

常见种类:
    线性核函数:
        k(x, z) = x · z
    多项式核函数:
        k(x, z) = (γ*x · z + r)**d   扩展到d阶     γ, r, d 为超参数
    高斯核函数:
        k(x, z) = e**(-γ||x-z||2**2)        γ 为超参数 需要大于0
    sigmoid核函数:
        k(x, z) = tanh(γ*x · z + r)       γ, r 是超参数, 需要调参

高斯核函数:
    将每一个样本点映射到一个无穷维的特征空间:   
    高斯RBF 内核:
        from skleran.svm import SVC
        from skleran.pipeline import Pipeline
        from sklearn.preprocessing improt StandarScaler
        from sklearn.metric import mean_squared_error
        
        poly_kernel_svm_clf = Pipline([
            ('scaler', StandarScaler()),
            ('svm_clf', SVC(kernel = 'rbf', gamma = 5, C =5))    # kernel 核函数
        ])
        poly_kernel_svm_clf.fit(X, y)
        y_predict = poly_kernel_svm_clf.predict(X)
        mean_squared_error(y, y_predict)

多元线性回归
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_moder import LinearRegression
    from sklearn.preprocessing import StandarScaler

    def PolynomialRegression(degree):
        return Pipeline([('poly', PolynomialFeatures(degree = degree)), # 添加多项式
                        ('std_scale', StandardScaler()),
                        ('lin_reg', LinearRegression())
                        ])

    poly_reg = PolynomialRegression(degree = 2)
    poly_reg.fit(X, y)
    Pipeline(memory = None, steps = [('poly', PolynomialFeatures(degree = 2, include_bias = True, interaction_only = False)), 
                                    ('std_scale', normalize = False)])
    y_predict = poly_reg.predict(X)
    mean_squared_error(y, y_predict)  均方误差

"""