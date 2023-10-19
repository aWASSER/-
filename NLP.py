# NLP(Natural Language Processing)

# 基础
"""
词向量:
    文本数据转换, 将文本转化为数据
    人工智能统称为Word Embediing, 是一种映射embedding, 把字/词映射成向量

方法:
    序号化(标签1, 2, 3), 哑编码(ont-hot)对每个下标进行转换, 词袋法(BOW/TF)对每个词唯一下标转换
        哑编码: 每个词代表一个向量, 有就是1 没有就是0 构成矩阵
        词袋法: 是哑编码的累加
    TF-IDF
        在词袋法基础上加入单词重要性影响系数, 计算单词在该情况下出现的次数
    主题模型(LSA, LDA等)
    word2vec(主要方式, 词向量), char2vec(字向量)
        word2vec:
            结构:
            input--> Embedding layer  -->  Output
                CBOW    : 基于上下文预测周边单词 
                Skip-gram: 基于中间的预测周边的
            思路:
                1. 每个词对应一个向量
                2. embedding, 做一个向量值的相加得到一个向量, 做一个全连接, --> 输出 [1, 10000] 表示当前样本属于10000个类别的置信度, 10000个类别就是单词数目
            词典的构建:
            CBOW(需要将文字转化为onehot):
            批次数目 * 单词数目(n) * 维度数目(128/64)  ===> embedding 是 [3 vim] * [3, 128]  ==> 3*128 (上下文环境特征) 
                        ===> FC全连接[128, 单词数目(n)] ==> [2, n(第一个单词预测为n的置信的度)] ==> 构造交叉熵损失函数
                import jieba
                text1 = "我爱我的国家"
                text2 = "我来自湖北张家界"
                a = jieba.lcut(text1)
                b = jieba.lcut(text2)
                c = set(a)
                for _b in b:
                    c.add(_b)
                words = ['我', '国家']
                wd = {w: i+1  for i, w in enumerate(c)}    # 得到一个词典
                wd['unk'] = 0     # 得到一个完整的词典
                word_dict = wd
                word_idx = [word_dict.get(w, word_dict['unk']) for w in words]    # 文本 --> 序号
                word_onehot = [[0]*len(word_dict), [0]*len(word_dict)]
                word_onehot[0][word_idx[0]]=1
                word_onehot[1][word_idx[1]]=1
                word_onehot
                交叉熵损失函数:
                p = e**zi / ∑ e**zi (i∈[1, n])
            Skit-gram:
                一个样本w(t)中间词 =向量化=> [n, 128]  => FC全连接[128, n] ==> 输出多个[2, n] 的矩阵
            目的: 
                1. 找一个置信度内一个最高的值
                2. 相同语义词向量相似/相同
        
        char2vec:
            和word2vec一样, 不进行分词, 直接将字符转换成字向量, 后续基于字向量建模
            分段后, 把一个词向量划分为几个字向量

    Doc2vec(文本向量)
        使用Word2vec 作为第一步输入, 然后利用word2vec的单词向量对每个句子或者段落生成复合向量
    FastText, cw2vec
        FastText: 输出文章中所有词
            只有一层隐层和输出层, 结构和CBOW类似
            区别:
                CBOW中输出的是词向量, FastText输出的是类别label
                CBOW中输入是当前窗口除中心外的所有词, 而FastText输入是文章所有词
            训练过程中增加了subwords特性 ----> 加入一个词的character-level 的 n-gram
            在分类中也增加了N-gram的特征, 主要为了通过增加N-Gram信息保留词序(隐层通过简单求和平均得到)
        cw2vec:
            基于偏旁部首进行
            步骤:
                1. 将词语分割成字符
                2. 提取笔画信息, 然后将所有字符笔画组合到一起
                3. 查表得每个笔画id组成这个词语对应的id
                4. 产生N-gram 笔画特征
    gensim(常用, 经典)

方式:
    通过矩阵乘法的得到一组向量, 神经网络将词表中的词语作为输入, 输出一个低维词向量表示这个词, 
    通过反向传播不断参数优化, 输出的低维词向量是神经网络第一层输出, 这一层称为Embedding Layer
    torch.nn.Embedding(词表大小, 维度)

霍夫曼树的构建:
    输入: 权重为(w1, w2....wn) 为n个节点
    输出: 对应霍夫曼树(二叉树)
    步骤:
        将(w1, w2.....wn) 看成n 棵树, 每棵树看成一个节点
        在森林中选取根节点最小的两棵树进行合并得到一个新的树, 新树根节点权重是两棵树权重之和
        将之前根节点最小的两棵树从森林删除, 并将新树加入
        重复步骤直到只有一棵树

负采样构建:
    关键思想在于二分类
    按频率抽取
    实际类别看为正例, 抽取非实际类别中随机抽取k个作为负例
    从所有非实际类别中随机的抽取k个类别作为负例, 然后进行线性变换, 最后更新损失函数以及反向更新参数

多分类评估标准:
    混淆矩阵:
        np.mean(re1, re2, re3)
        recall_score(y_true, y_pred, average='macro')
    ROC_AUC曲线:
        roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
"""

# NLP 自然语言处理
"""
NLP 自然语言处理
目的:
    让计算机处理或理解自然语言, 以执行语言翻译和问题解答等任务, 最终来讲就是构建机器语言和人类语言中间的桥梁, 实现人机交流为最终目的

常见应用:
    · 关键词提取, 概要抽取, 事件抽取
    · 命名实体识别(提取价格, 日期, 姓名, 公司等)
    · 关系抽取
    · 分类: 文本分类, 情感分析. 意图识别
    · 机器翻译
    · 语音文本转换, 图像文本转换
    · 问答系统

NLP分为 NLU 和 NLG 两个子任务:
    NLU: 自然语言理解
    含义: 让计算机理解自然语言文本的意义
    目标: 分词, 词性标注, 文本分类, 实体识别, 信息抽取等

    NLG: 自然语言生成
    含义: 将非语言模式转化为人类可以理解的语言
    目标: 机器翻译, 问答系统, 聊天机器人
"""

# 神经网络
"""
BP神经网络就是神经元的堆叠

构成:
    输入层 --- 隐藏层(神经元, 隐层) --(激活函数)-- 输出层

来源: 大脑神经元细胞模拟, 树突接受电信号, 信号通过细胞核处理后将处理信号传给下一个神经元
      一个神经元可以看为一个或多个输入处理成一个输出的计算单元
      通过多个神经元传递最终大脑会得到这个信息, 并进行反馈

感知器模型 / Logistic回归
感知器是一种模拟人的神经元的一种算法模型, 是一种研究单个训练样本的二分类器, 是SVM 和人工神经网络(ANN)的基础
一个感知器接受几个二进制的输入, 并产生一个二进制输出, 
   output = | 0    if ∑ wj * xj <= threshold  or  if w * x + b <= 0
            | 1    if ∑ wj * xj > threshold  or  if w * x + b > 0

softmax回归是logistic 回归的一般化, 适用于k 分类
可以使用L2正则, dropout和输入噪声等抑制过拟合

神经网络公式:
    输入: x1, x2, x3 和截距 +1 
    输出: hw,b(X), 其中w 为权重, b 为偏置向参数
    hw, b(X) = f(W.T * x, b) = f(∑ Wi * xi + b)

激活函数:
    作用:
        提供网络的非线性模拟能力, 加入激活函数后, 深度神经网络才具备了分层的非线性映射学习能力.
        提升了神经网络模型的表达能力
    特征:
        可微性, 单调性, 输出值的可控范围
    常见的激活函数: 
        Sign 函数, Sigmoid 函数, Tanh (RNN中作为激活函数)函数, ReLU (修正性线性单元)函数, P-ReLU 函数, Leaky-ReLU 函数, ELU 函数
        二分类: sigmoid
        多分类: sotfmax
    sigmoid(二分类):
        import numpy as np
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
    Relu(梯度下降快, 计算快):
        import numpy as np
        def _relu(X):
            return np.maximum(0, X)
    softmax(多分类):
        import numpy as np
        def _softmax(x):
            c = np.max(x)
            exp_x = np.exp(x - c)
            return exp_x / np.sum(exp_x)

前向传播:
    全连接, 训练权重(wi), 达到隐藏层训练权重(wj), 最后得到 输出值
    权重初始化: 对于由特征向节点(隐层)得到的权重
    weights: 根据线有关, 有几个特征就是 [特征数, 第一个层节点数] 
    bias:  [第一层节点, ], [第二层节点, ]

输出神经元的个数(输出层):
    一般和类别数量保持一致

损失函数:
    分类: crossentropy
    回归: MSE   

浅层神经网络:
    添加少量隐层的神经网络叫做浅层神经网络, 也叫传统神经网络, 一般为2 隐层神经网络.
深层神经网络:
    增多中间层(隐层)的神经网络称为深层神经网络(DNN), 也叫全连接神经网络
神经网络非线性可分:
    对线性分类器的与或和的组合可以完成非线性可分的问题, 通过多层的神经网络中加入激活函数的方式可以解决非线性可分的问题
隐层与决策域:
    无隐层: 由一个超平面分成两个
    单隐层: 开凸区间或闭凹区间
    双隐层: 任意形状, 复杂度由单元数决定

神经网络过拟合:
    理论: 神经网络可以逼近任何连续函数
    效果: 网络工程效果中, 多层神经网络比单层神经网络效果好
    分类: 三层神经网络优于两层神经网络, 再往上加效果不会太好
    表达: 提升隐层层数或神经元个数, 神经网络容量变大, 表达能力变强, 从而可能导致过拟合
    不利: 视频/图片识别 传统神经网络(全连接神经网络)不太适合

梯度下降法:
    常用于求解无约束情况下凸函数的极小值, 是一种迭代算法.

BP神经网络
正向传播(FP)求损失: 往前算梯度, X和矩阵V乘法 → 再进行激活转换 → 得到一个输出 → 得到一个 loss, 
反向传播(BP)回传误差, 根据误差值修改每层权重, 继续迭代 改变参数 w v 得到一个新的过程

输出层误差:
    E = 1/2 * (d - O)**2 = 1/2 * ∑(dk - Ok)**2     k∈[1, t]
隐层误差:
    E = 1/2 * ∑(dk - f(netk))**2 = 1/2 * ∑(dk - f(∑wjk * yj))**2 
输入层误差:
    E = 1/2 * ∑(dk - f[∑wjk * f(netj)])**2  = 1/2 * (dk - f[∑wjk * f(∑ vij * xi)])

简单的三层神经网络

"""

# PyTorch基础
"""
核心概念:
    张量: Tensor对象
    自动求导数: autograd
import torch
torch.rand(int a, int b)    # 随机 均匀分布
torch.rand_liked()    # 参考别的模仿
torch.randn()       # 产生服从高斯分布的值
torch.arange(0, 10, 3)    # 步长为3 的范围数据
torch.linespace(0, 10, 4)    # 0- 10 线性范围
torch.eye(4)    # 创建一个 单位矩阵
a = troch.empty(2, 3)    # 创建一个两行三列的空矩阵 空有可能是0 有可能是其他值
a.fill_(0.0)     # 加了下划线表示直接在 Tensor 对象上进行操作, 没有加则是产生新的值
a.normal_()    # 正态分布初始化
a.add(int)     # 矩阵内每一个数加一个int 数据
torch.dot()    # 内积
torch.matmul(c, d)    # 矩阵乘法
torch.T     # 转置
torch.view((3, 2))  # 转为 3行2列
torch.rand(2, requires_grad = True)      #  requires_grad 是否需要计算梯度的值

x * w + b
x_ = torch.rand(3, 2)
w_ = torch.rand(2, 5)
b_ = torch.rand(5)
y_ = x_ @ w_ + b_

torch.hsplit      # 分割
"""

# 代码部分:
from operator import mod
import torch
import numpy

from bp4 import Network

""" 前向过程 """
a = torch.tensor([2., 3.], requires_grad = False)
# 还可以这样:   requires_gard : Pytorch 中控制是否需要计算梯度的参数值, 默认为False 后面的不进行梯度, 前面的也不进行梯度计算
# a = torch.rand(2, requires_gard = True)
b = torch.tensor([6., 5.], requires_grad = True)
c = torch.sum(a * b)
d = 20 - c
loss = d ** 2

# 进行计算
print("-" * 50)
d1 = torch.sum(a * b)
d2 = torch.sum(a * c)
print("d1:", d1)
print("d2:", d2)
print("-" * 50)
e1 = d1 - 20
e2 = d1 - 30
print("e1:", e1)
print("e2:", e2)
print("-" * 50)
loss = e1**2 + e2**2        # 这里也可以加 detach
print("loss:" , loss)
print("=" * 50)

for i in range(10):
    d1 = torch.sum(a * b)
    d2 = torch.sum(a * c)
    e1 = d1 - 20
    e2 = d1 - 30
    _e1 = d1 - 20  # _e1 先调用类里面 __call__方法, 然后在__call__里面实现, 再调用下面函数 直接调用类里面内容
    if i % 2 == 0:
        e1 = e1.detach()   # 不更新t1  反向时候 detach() 梯度终止
    else:
        e2 = e2.detach()   # 不更新t2
    loss = e1 ** 2 + e2 ** 2
    print(loss)
    print("==" *50)
""" 反向过程 """
print(b.grad)  # 没有梯度 是 None
# backend   (只负责计算梯度)每执行一次, 将当前loss对于参数的梯度累加到对应参数的grad 属性上  -->  实际上PyTorch框架会做一个控制, 不允许累加
loss.backward()
print(b.grad)
learning_rate = 0.01
print("更新前:", b)
b.data.sub_(b.grad.data * learning_rate)      # 更多参数b
print("更新后:", b)

# 卷积神经网络(CNN)
"""
CNN更多应用于图像识别

通过三原色进行体现像素 长*宽*高*颜色数 ---> 体现出一个图形
例子:
    像素 800 * 500 * 3 = 120w

CNN 中新增了 Convolution 层和pooling 层, 卷积层(Convolution) 与全连接层不同, 卷积层不改变特征数据结构
局部感知: 将图片划分成一个个区域进行计算
参数共享: 每个神经元卷积核连接数据共享

卷积层: 
    目的 ----> 提取特征     输出数据称为输出特征图

卷积运算: 
    对图像处理, 让卷积核在输入特征图上进行滑动, 从左到右从上到下, 每滑动一次进行一次卷积运算
    原图特征 * 卷积核 = 结果
    运算方式, 横乘横

卷积核(学习训练得来): 
    卷积大小在实际需要时需要定义其长和宽, 其通道个数与图片通道个数相同

featrue map 计算方式:
    横着算 横向 * 横向
    每个区域会形成一个feature map
    之后进行池化, 形成一个值

filter:
    每次取的单词长度和词向量长度 

填充:  
    输出大小 = 1 + (输入大小 + 2*填充大小 - 卷积核大小) / 步长大小
    做一个像素的填充(padding = 1)
    ex: 对一个步长为1, kernel 为3*3 的卷积输出的特征层则为7*7 输入图像为: w1 * H1 * D1 长*宽*高
        卷积层中 kernel 大小为F*F, kernel个数为k, 步长S, padding 大小为P, 经过卷积输出图像长宽高分别为:
            w2 = (w1 - F + 2P) / S + 1
            H2 = (H1 - F + 2p) / S + 1
            D2 = k
            里面有多少个参数进行运算: 
                权重wi:F*F*3 * k
                bias: k

conv2d:
    二维卷积: nn.conv2d, 对宽度和高度都进行了卷积运算
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    in_channels: 输入通道数 ----> feature map 数目
    out_channels: 输出通道数 ----> 卷积核数目
    kernel_size: 卷积核大小---> 窗口大小
    stride: 卷积核滑动窗口移动步长
    padding: 填充大小
    self.convl = nn.Conv2d(1, 6, 5)    # 输入通道数为1, 输出通道数为6, 卷积核为5*5

池化层:
    对图片进行压缩, 减少计算量, 不会改变特征提取
    方法:
        max pooling
        average pooling
    参数: 
        池化kernel 大小为F*F, 步长为S, 输出图形的宽和高为:  
            w2 = (w1 - F) / S + 1
            H2 = (H1 - F) / S + 1
            D2 = D1
    特点:
        没有要学习的参数, 只是从目标区域总取最大值或者平均值

LeNet:
    解决手写数字识别视觉任务
    LeNet-5 表示有5 个层
    激活函数: 
        tanh, ReLu
    
Textcnn:
步骤:
    1. 输入input
    2. 提取区域. filter: 每次取的单词长度和词向量长度 
    3. 提取featrue map, 把词向量变为[所选单词长度, 1]
    4. 池化: 对feature map进行压缩, 整个区域算一个值
    5. 之后对各段进行拼接
    6. 最后进行softmax, fc(全连接)
提取文本相邻特征单词的特征信息
torch.nn.Conv2d + 特殊卷积核
本来的NLP的数据shape形状为[N, T, E] 如果用CNN, 需要将数据转化为[N, 1, T, E] --> [N, C, H, W ]

"""

# RNN 循环神经网络
"""
RNN结构:
    表示:
        input: x1[i1, i2, i3]: 1x3的矩阵   (可以做word2vec --> 可以做embedding)
        ↓ 得到当前时刻的细胞信息
        h1 = sigmoid(x1 * w + b(偏置项) + h0*u)     参数:(x1: 1x3矩阵, w: 3x4矩阵, b: 1, h0: 1x4矩阵, u: 4x4矩阵)   这里的u的矩阵是看hidden里面有多少层, 同w的4 和 h0的4
        h1 = sigmoid([x1, h0]*w + b)     这个需要把x1 和 h0 列concat(拼接)
        ↓
        然后在State 保存(记忆) RNNCell
        ↓ 得到细胞信息后和下一个合并
        h2 = sigmoid(x2 * w + b + h1*u)         由于相同情景, 所以w是一样的, 同样b和u也是一样的, 是共享的

    参数:
        x = [x1, x2, x3, ..., xt]       xt 代表一个词向量, 一整个序列(x) 就是一句话
        ht: 代表
        ot: 表示t时刻的输出
        输入层到隐藏层之间的权重用 u 表示, 它将我们的原始层输入进行抽样作为隐藏层的输入
        隐藏层到隐藏层的权重 w, 是网络记忆的控制者, 负责记忆调度
        隐藏层到输出层的权重 V, 是隐藏层学习到的表示将通过再一次抽象, 作为输出结果

    思路:
        将序列按时间展开就可以得到RNN结构
        xt: 时间在t处的输入
        st: 时间t处的记忆, st = f(uxt + Wst-1), f可以是非线性转换函数, 例如: tanh
        ot: 时间t处的输出, 比如是预测下一个词, 可能是sigmoid/softmax输出的属于每个候选词的概率, ot = softmax(Vst)

RNN正向传播:
    在t1=1时刻, u, v, w都被随机初始化, h0通常初始化为0, 然后计算:
        s1 = u*x1 + w*h0
        h1 = f(s1)
        o1 = g(v*h1)
    时间向前推进, 当时间为h1时, 记忆状态下参与下一次预测活动:
        st = u*xt + w*ht-1
        ht = f(st)
        ot = g(v*ht)

RNN反向传播:
    BP神经网络用到的 误差反向传播方法 将 输出层的误差总和 对各个权重梯度▽u, ▽v, ▽w, 求偏导, 然后利用梯度下降方法更新每个权重
    对于每一个时刻t的RNN网络, 网络的输出ot都会产生一定的误差et, 误差的损失函数, 可以是交叉熵也可以是平方误差
    总误差: Et = Σ et
        ▽u = δEt / δu = δΣ et / δu
        ▽v = δEt / δv = δΣ et / δv
        ▽w = δEt / δw = δΣ et / δw
    对于链式求导后进行梯度下降, 不断更新参数u, v, w 
    为了克服梯度消失问题, LSTM 和 GRU 可以一定程度上克服梯度消失, 对于RNN来言, 很早之前时刻输入的信息, 对当前时刻不影响
    为了克服梯度爆炸问题, gradient clipping 可以一定程度上克服梯度爆炸, 当计算梯度超出阈值[-C, C]时, 便把梯度设置成-C或者C
    
代码:
    import torch
    from torch import nn
    rnn = nn.RNN(10, 20, 2)     # 10 是inputsize 输入特征向量的维度大小, 20 是hiddensize里面每个圆圈, 一个特征连20个圆圈--用几维的向量表示特征信息  2 是有几层, 层数大小
    # list(rnn.parameters())      # 查看数据结构
    # for w in list(rnn.parameters()):    # 查看数据尺寸
    #     print(w.shape)
    input = torch.randn(5, 3, 10)   # 5是样本, 3是时刻--每个时刻是10维向量 ↑ 上面inputsize
    h0 = torch.randn(2, 3, 20)      # 定义h0
    output, hn_hiddensize = rnn(input, h0)
    output.shape
    hn_hiddensize.shape

双向RNN(Bidirectional RNN):
    t 的输出不仅仅和之前序列有关, 还和之后序列有关, 所以由两个RNNs 上下叠加组合在一起, 输出由这两个RNNs的隐层状态决定.

Deep RNN(深度双向RNN):
    在双向RNN的基础上, 多添加添加多层网络结构, 加强表达能力和学习能力, 复杂性提高, 训练数增多

RNN-BPTT:
    随机梯度下降算法中, 每一步不仅仅需要依赖当前步的网络, 还需要之前的网络状态

LSTM:
    LSTM 对记忆细胞进行改造, 对应该记忆的信息会一直传递, 不该记忆的信息会截断掉
    由参数决定是否截断, 也就是RNN反向更新决定
    细胞状态信息是细胞状态信息的线, 细胞输出是细胞输出的线
    1. 细胞状态---存放的是长期记忆
    2. 隐层状态---存放的是短期记忆
    关键:
        细胞状态, 类似于传送带, 直接在整个链上运行, 信息在链上不容易变
    如何控制细胞状态:
        ·LSTM 通过控制门(gates, 符号: σ)结构来去除或者增加细胞状态的信息
        ·包含一个sigmoid神经网络层次和一个pointwist乘法操作
        ·sigmoid 层输出一个0-1的概率值, 描述有多少个量可以通过, 0表示不允许通过, 1表示运行所有变量通过
        ·LSTM 中主要有三个"门" 结构来控制细胞状态
            三门:
                1. 遗忘门, 信息删除, 信息丢弃
                    对于上一个时刻信息丢弃
                    ft = σ(Wf * [ht-1, xt] + bf)
                    假设一个时刻输入样本xt 有 5 个样本(batch大小)表示为[T(所有时刻), 5, 10], 每个样本有十维向量, ht-1 表示 有5个样本, 有20维向量 输出大小也为[5, 20]
                    batch 与 output 拼接后 输出 [5, 30]
                2. 增加门, 信息加强
                    决定哪些信息需要放在细胞中间
                    对于现在时刻信息加强
                    it = σ(Wi * [ht-1, xt] + bi)          取值[0, 1]
                    C^t = tanh(Wc * [ht-1, xt] + b^c)    [5, 20] 取值[-1, 1]
                    Ct = ft * Ct-1 + it * C^t
                3. 输出门, 信息输出
                    运行sigmoid 层确定细胞状态的那个部分输出
                    使用tanh处理细胞状态得到[-1, 1]之间的值, 再将它和sigmoid门的输出相乘, 输出程序确定输出部分
                    ot = σ(Wo * [ht-1, xt] + bo 
                    ht = ot * tanh(Ct)    
    LSTM变种:
        1. 增加 peephole connections层
            让层门接受细胞状态的输入 一个门连到另一个门上
            ft = σ(Wf * [ht-1, xt] + bf)
            it = σ(Wi * [ht-1, xt] + bi) 
            ot = σ(Wo * [ht-1, xt] + bo)
        2. 通过耦合遗忘门和更新输出门
            不再单独考虑忘记什么, 增加什么信息, 而是一起考虑
            Ct = ft * Ct-1 + (1 - ft) * C^t 

GRU:
    输出门与遗忘门结合生成了单独的更新门, 还合并了细胞状态和隐藏状态--->细胞状态和输出状态是同一个值
    只有一种因此状态, 依靠门的机制和运算存储短期和长期记忆
    zt = σ(Wz * [ht-1, xt])    遗忘门和更新门
    rt = σ(Wr * [ht-1, xt])    决定输出时, 在上一个时刻对现在有多大影响
    h^t = tanh(W * [rt * ht-1, xt])
    ht = (1 - zt)*ht-1 + zt * h^t

输入数据:
    input: [seq_len, batch, input_size]
    h0: [num_layers*num_directions, batch, hidden_size]

输出数据:
    output: [seq_len, batch, num_directions*hidden_size]
    hn: [num_layers*num_directions, bathc, hidden_size]
代码:
    import torch
    import torch.nn as nn
    rnn = nn.GRU(6, 5, 1)  # GRU(embedding_size, hidden_dim, num_layers)
    inputs = torch.randn(3, 1, 6)   # 模拟文章词向量, (seq_len, batch, embedding_size)
    output, last_hidden = rnn(inputs)
    final_output = output[-1, :, :]
    final_hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), 1)
    linear = nn.linear(10, 2)    # 全连接层   # 2是二分类
    temp_predict = linear(final_hn)
    classifier = nn.sotfmax(dim = 1)    # 我们对类别做softmax
    predict = calssifier(temp_predict)

""" 

# gensim代码:
import gensim
from gensim.test.utils import common_texts
from gensim.models import TfidfModel, LdaModel, LsiModel
from gensim.corpora import Dictionary
# 数据加载
with open('路径') as reader:
    content = reader.read()
# 单词划分, 转换成二进制
words = list(map(lambda word: word.encode("utf-8"), filter(lambda t: t.strip(), content.split(""))))
total_words = len(words)
print(f"单词数目: {total_words}")
print(f"[前十个单词]: {words[0: 10]}")
# 将其转换为文档形式(必须步骤, 一个文档存在多个单词) --> 模拟多个文档
word_pre_doc = 10000
docs = []
for i in range(total_words // word_pre_doc + 1):
    # 获取索引
    start_idx = i * word_pre_doc
    end_idx = start_idx + word_pre_doc
    # 获取对应单词表
    tmp_words = words[start_idx, end_idx]
    # 保存
    if len(tmp_words) > 0:
        docs.append(tmp_words)
print(f'文档总数目为: {len(docs)}')
# 构建字典
dic = Dictionary(docs)
# BOW词袋法转换
corpus = [dic.doc2bow(line) for line in docs]
# TFIDF构建
model = TfidfModel(corpuse = corpus)
# TFIDF应用--预测
pre = model[corpus[0]]
for i in pre:
    print(i)

# LDA模型--->模型构建
from gensim.models import TfidfModel, LdaModel, LsiModel
model = LdaModel(common_corpus, num_topic=10)
# 模型保存
model.save('路径')
# 模型加载
lda = LdaModel.load('路径')
# 模型路径获取(文本向量)
vectors = lda[other_corpus]
for vector in vectors:
    print(vector)
# 更新模型
lda.update(other_corpus)
# 更新后文本提取
vectors = lda[other_corpus]
for vector in vectors:
    print(vector)
# 各个单词对应的主题向量
model.get_topics().shape

# word2vec
from gensim.models import word2vec
import jieba
import jieba.analyse
import numpy as np
import gensim
# 数据加载
content = reader.read()
# 分词
content = jieba.cut(content)
# 结果输出
Writer.write(result)
# 数据加载
sentence = word2vec.LineSentence('路径')
# 训练word2vec模型
model = word2vec.Word2Vec(sentence, hs = 1, min_count = 1, window = 3, vector_size = 100 ,sg = 0)   # sg: 1(skip gram) 0(CBOW)  hs: 1(哈弗曼树) 0(负采样)
# 构建词典
model.build_vocab(sentence)
# 模型训练
model.train(sentence, total_examples = model.corpus_count, epochs = 5)    
# 获取相关属性
print(f'词汇数目: {len(model.wv.key_to_index)}')
print(f'转换的稠密特征向量维度: {len(model.wv.vector_size)}')
print(f'单词到id的映射关系: {model.wv.key_to_index}')
# 夹角余弦相似度
req_count = 5
for key in model.wv.similar_by_word('名字', topn = '100'):    # 先找到名字, 然后在文本向量里面找最相似的
    if len(key[0]) == 3: 
        req_count = -1
        print(key[0], key[1])
        if req_count == 0:
            break
# 获取余弦相似度
print(model.wv.similarity("A", "B"))
# 获取单词词向量
v1 = model.wv.get_vector('某个词')
print(v1.shape)
print(v1)
# 使用numpy api 保存词向量 ---> 持久化
norm_word_embedding = model.wv.vectors_norm
word_embedding = model.wv.vectors
# 获取词典向量(词典到index的映射)
vocab_2_index = list(map(lambda k: (k, model.wv.key_to_index[k]), model.wv.vocab))
# 数据保存
np.save(model_file_path3.format('norm_word_embedding'), norm_word_embedding)
np.save(model_file_path3.format('word_embedding'), word_embedding)
np.save(model_file_path3.format('vocab_2_index'), vocab_2_index)
# 数据加载
norm_word_embedding = np.load(model_file_path3.format("norm_word_embedding"))
word_embedding = np.load(model_file_path3.format("word_embedding"))
vocab_2_index = np.load(model_file_path3.format("vocab_2_index"))
# 字典转换
vocab_2_index = dict(map(lambda t: (t[0], int(t[1])), vocab_2_index))
# 获取数据
word = '要获取的内容'
index = vocab_2_index[word]
v1 = word_embedding[index]
print(v1.shape)

"""
seq2seq 是从时序到时序的意思, 即将一个时序数据转换为另一个时序数据
结构:
    通过组合两个RNN, 实现seq2seq
    模型有两个模块:
        1. Encoder(编码器): 将文字转化为二进制
        2. Decoder(解码器): 将二进制转化为文字
一般我们使用RNN来处理变长的序列

在输入和输出两个独立序列中间加入中间容器----语义编码层
中间容器(c) 包含了所有原数据信息, c 拿着数据进行训练
C左边的直接用了RNN-GRU  -----> 只需要在意是否传到容器C内-----Encoder
C: context ----> last_hidden
C右边的用了for循环  ------> 需要关注输出的内容-----Decoder
特点:
    1. 不论输入和输出的序列长度是怎样的, 中间语义编码层c的向量长度固定
    2. 不同任务中的编码器和解码器是可以用不同算法来实现的, 一般是RNN系列中的一种

SRC.vocab.itos[矩阵里的数字]   # 查看数字对应的字
"""

# bleu-评估翻译模型指标
"""
机器翻译: 一般为predict
人工翻译: 正确答案 --> reference
第一步: 计算各阶n-gram 精度
    概率 = 分子减2 / 分母-1    分子 > 0
第二步: 加权求和
    wn = 1 / len(上面计算的概率数量)

惩罚因子:
    避免翻译部分句子很精准, 但是匹配度依然很高的问题, 所以引入长度惩罚因子
        BP = 1 if c > r else exp(1-r/c)     c 为机器翻译结果长度 , r 为人工翻译结果长度   
第三步: 求BP
    机器翻译长度 = 参考译文长度    <==>   BP=1
    BLEU-4 = ∑ wn * log(pn)

Sentence BLEU score
    from nltk.translate.bleu_socre import sentence_bleu
    reference = [['this', 'is', 'a', 'test'], ['this', 'is', 'test']] 
    condidate = ['this', 'is', 'a', 'test']
    score = sentence_bleu(reference, condidate)
"""

# attention
"""
weight: αi = align(hi, s0)  ----- 权重
    方法1(simpleRNN + Attention):
        第一步: 算出 α^i:
            a^i = V.T * tanh(W * [1, len(texy[i][i])词向量长度])   ----->  1. hi  2. S0
        第二部: 算出 αi:
            αi = softmax(a^i)
    方法2:
        1. Liner maps:
            ki = wk * hi, for i = 1 to m
            q0 = wq * s0
        2. inner product:
            a^i = ki.T * q0, for i = 1 to m
        3. Normalization:
            αi = softmax(a^i)

Context vector: c0 = α1 * h1 + ···· + αm * hm   ------  语义层
    c0 是包含了所有encoder状态的加权平均, 所以s1的计算依赖于c0, 解决了之前内容被遗忘的问题
    c1 = α1 * h1 + ···· + αm * hm   ----  重新计算_> 值会不一样 与c0关注点不一样
    s0 = encoder(last_hidden)
    s1 = tanh(A`*[x`1, s0, c0]+b )
    这里的 c1 c2 c3 ----- cn 就是attention
    s2 = tanh(A`*[x`2, s1, c1]+b )
    问题: 如何在python中实现点乘, 1. torch.nn (权重需要定义)    2. nn.Linear(dec_hid_dim, 1, bias = False)点乘
    数据形状变换:
        1. unsqueeze(需要加的位置下标)
        2. hidden.repeat(scr_len, 1, 1)     # [src_len, batch_size, enc_last_hidden_dim]
    attention = [batch_size, src_len]

bmm函数:
    计算两个tensor 之间乘积, 要求两个tensor是三维
"""

# transformers
"""
输入embedding后, 一部分变为Multi-Headed Attention输入 一部分变为Multi-Headed Attention输出, 之后进行Norm, 再放到前向网络, 再进行Norm
input embeddings
class embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forword(self, x):
        return self.embed(x)

self-attention
    q: query(to match others)
        qi = Wq*αi
    k: key(to be matched)
        ki = Wk*αi
    v: information to be extracted
        vi = Wv*αi
    顺序: x1(embedding) --> α1 ---> q1, k1, v1
        α1 = W*x1
    接下来:
        自己的q 和其他的关联
        拿每个 query q 去对每个 key k 做 attention  
        α1,i = q1 * ki / d**(1/2)        q1 * ki ---> dot product    d ---> q 和 k 的 dim
    softmax:
        得到的结果  ∑(α1, i) = 1
        α^1, i = exp(α1, i) / ∑(exp(α1, i))
    总体sequence(顺序):
        各个词和其他的关系
        b1 = ∑(α1, i) * vi
        同理 b2 = ∑(α2, i) * vi

Multi-head self-attention 多头self-attention
    将self-attention里的 qi, ki, vi 在得到后拆成两份
    qi ---> (qi,1), (qi,2) 拆成几份就是几头, 这里是两头
    qi,1 = wi,1 * qi
    qi,2 = wi,2 * qi
    关联的时候只需要和相应的i, j 的j 关联即可
    之后需要concat 几个头相加
    concat 后进行linear

output embedding
    分类类别: (batch_size, 1) 即 target向量
"""

# Positional Encoding
"""
位置encoding, 讲原先含有的权重W 变为 w1, w2, 分别乘xi pi
    得: w1 * xi = αi
        w2 * pi = ei   ---> 位置权重
        随着反向传播可以不更新
相当于在 αi 处加入位置权重 ei
positional = torch.arange(0, seq_length).expend(N, seq_length).to(self.device)
out = self.dropout(self.word_embedding + self.positional_embedding(positional))
"""

# Residual 残差
"""
有一个 F 是求和前网络, H 是求和后网络, x=5 映射到5.1, 残差为F`(5)=5.1, H`=F` 也是5.1, 如果H从5.1 变为 5.2 映射F` 的输出增加1.96%
引入残差后对输出变化更敏感----思想: 去除相同主体部分, 突出细小的变化
"""

# LayerNorm
"""
batchNorm 和 LayerNorm 区别:
    batchNorm 是横着的数据---数据个数
    LayerNorm 是竖着的数据---是一个数据的维度
"""

# mask
"""
padding mask:
    每个批次输入长度不一样, 对输入长度进行对齐, 在短的后面填充0, 输入长的把多余的直接舍弃
sequence mask:
    为了使decoder不能看见未来的信息, 使t时刻解码输出只能依赖于t时刻之前输出(looK ahead mask)
masked_fill:
    scores.masked_kill(mask==0, -1e9)
    用于把mask是0的变成一个很小的数, 后面经过softmax得到概率接近0的值
    代码:
        def make_src_mask(self, src):
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            return src_mask.to(self.device)
        def make_trg_mask(self, trg):
            N, trg_len = trg.shape
            trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
            return trg_mask.to(self.device)
权重初始化:
    def initalize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    model.apply(initalize_weiights)
"""

# 学习率调整cheduler
"""
1.torch.optim.lr_scheduler.StepLR
    torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma = 0.1, last_epoch = -1)
    功能: 等间隔调整学习率, 调整倍数为gamma 倍, 间隔为 step_size, 间隔单位为 step(epoch)
    思路:
        如果 间隔单位(epoch) / step_size(间隔) = 0  则学习率(lr)变为 gamma*lr(epoch-1) 

2. lr_scheduler.MultiStepLR
    torch.optim.lr_scheduler.MultiStepLR(optimizer, step_size, gamma = 0.1, last_epoch = -1)
    功能: 适合后期调用的 按设定间隔调整学习率
    思路:
        如果 last_epoch 在 optimizer 里 改变学习率

3. lr_scheduler.ExponentiallLR
    torch.optim.lr_scheduler.ExponentiallLR(optimizer, gamma = 0.1, last_epoch = -1)
    功能: 按指数衰减调整学习率
    公式: lr = lr * gamma**epoch
    思路:
        当 last_epoch = -1 时 改变学习率

4. LAMBDALR
    torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    功能: 当某个指标不再变化时, 调整学习率
    思路:
        lr = base_lr * Lambda(epoch)
"""

# 激活函数
"""
解决非线性可分问题
想法:
    需要根据每一层前面的激活、权重和偏置, 为下一层的每个激活算一个值, 在将该值发送给下一层前用激活函数对这个输出值进行缩放

sigmoid(二分类, 逻辑回归): 缩放为 0-1 之间,  需要求导数, 输入w*x + b 可能得到接近0 的值, 归一化能解决在有效值之内的缩放
    导数: e**(-x) / (e**(-x) + 1)**2
    梯度消失:
        梯度向量所有值都接近0, 无法更新任何东西
    梯度爆炸:
        所有wj 都很大 (wj > 1), 权重与权重缩放更新没有意义

ReLu的导数: x<0, y=0; x>0, y=y
    死亡ReLu: 导数为0, 0*learn_rate都为0, 很多值都是0, 所以会得到相当多不会更新的权重和偏置, 因为更新量为0
            bias_new = bias - learn_rate * gard(梯度)       # 梯度为0 无法更新

ELU(指数线性单元):
    思路:
        x > 0, ELU(x) = x; x < 0, ELU(x) = α * (e**x - 1)     # 这里α值需要选取
    导数:
        x > 0, 1; x < 0, ELU(X) + α
"""

# 优化器
"""
随机梯度下降(SGD)优化:
momentum:
    torch.optim.SGD(net.parameters(), lr=lr, momentum = 0.9, weight_decay = wd)
    # momentum = 0.9 随着梯度改变而改变
    更新改变:
        v_new = α*Vk-1 + (1-α)*g   这里的g就是梯度, α 是 momentum

adagrad(学习率自适应):
    设置步长, 限制了更新速度:
        rk = rk-1 + g·g
    更新参数:
        Θk = Θk-1 - ε / (γk + σ)**1/2 * g
    当比较大时, 分母比较大, 更新速度慢, 步长小
    当比较小时, 分母比较小, 更新速度快, 步长大
    adagrad不断增加,导致分母不断变大, 最后不更新

RMSProp:
    对adagard进一步优化
    rt = p*rk-1 + (1-p)*g·g    p 一般为 0.999 衰减次数 

adam:
    缝合怪, 融合了动量自适应避免了冷启动问题
    融合了: momentum 和 RMSProp两种
    累计梯度: vt = α*Vk-1 + (1-α)*g 
    累计平方梯度: rt = p*rk-1 + (1-p)*g·g 
    修正: 
        vt^ = vt / (1-α)
        rt^ = rt / (1-p)
    修正后可以设置一个较大的步长
    import torch
    torch.optim.adam(net.parameterss(), lr = learning_rate, momentum = 0.9)    # net.parameterss网络参数
"""

# 文本处理
"""
nltk内容:
将字母转化为小写:
exp: 
    input_str = 'The 5 biggest countries by population in 2017 are China, India, Untied States, Indonesia, and Brazil'
    input_str.lower()

删除文本中出现的数字:
    import re
    re.sub(r' \d+', '', input_str)      # 正则里多放一个空格, 这样就可以去除多余空格了
    re.sub(r'\d+', 'num', input_str)

删除文本中的标点:
中文标点写个正则reg = [，！。? "" 【】、]
    import string
    table = input_str.maketrans('', '', string.punctuation)     # string.punctuation 只能表示英文标点
    res = input_str.translate(table)

删除文本中的空格:
    input_str = input_str.strip()
    
删除文本中出现的终止词:
    文件预备:
        import nltk
        nltk.download('stopwords')
        nltk.download('wordsnet')
        nltk.download('punkt')

    from nltk.corpus import stopwords
    input_str = 'NLTK is a leading platform for building Python programs to work with human language data.'
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(input_str)
    res = [i for i in tokens if i not in stopwords]

词干提取(stemming)
    将词语简化为词干, 词根和词形的过程
    方式:   1. Porter stemming    2. Lancaster stemming
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    stemmer = PorterStemmer()
    input_str = 'There are several types of stemming algorithms.'
    input_str = word_tokenize(input_str)
    res = []
    for word in input_str:
        res.append(stemmer.stem(word))

词形还原(Lemmatization)
    变成了一个一个的
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lemmatizer = WordNetLemmatizer()
    input_str = 'been had done languages cities mice'
    input_str_stoken = word_tokenize(input_str)
    for word in input_str_stoken:
        print(lemmatizer.lemmatize(word))

spacy内容:
内容准备:
    python -m spacy download en_core_web_sm

代码:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u'This is a sentence and had a lot of books drove went mice.')
    tokens = [token for token in doc]

    # 一对一进行还原(时态, 复数等进行还原)
    for token in doc:
        print(token, token.lemma_)

    # 把原来的单词变成列表
    lem = [token.lemma_, for token in doc]

jieba内容:
    import jieba
    jieba.cut(X, cut_all = True)   # 切割字符, 分开字符  全模式   出现的可能性全部展示
    jieba.cut(X)                   # 精确模式
    jieba.cut_for_search(X)        # 搜索模式
    jieba.lcut(X)                  # 切好后的结果返回一个列表
    jieba.del_word("内容")         # 在后续中不会认为是一个单词
    jieba.add_word("内容")         # 在后续中, 认为是一个单词
    jieba.load_userdict("文件路径")    # 导入自定义词典库
    "/".join(jieba.cut("内容", HMM = False/True))     # 关闭/开启HMM发现新词功能   隐马尔可夫模型
    jieba.suggest_freq(("内容"), True)      # 建议怎么分开

    import jieba.analyse
    jieba.analyse.extract_tags(X, topK = 5, withweight = True)    # 关键词提取, 获取TF-IDF最大的五个单词    基于 TF-IDF 和 TextRank 算法

    # TextRank
    for x, w in anls.textrank(s, withWeight = True):
        print(f'{x}, {w}')

词性标注:
    import jieba.posseg as psg
    res = [(x.word, x.flag) for x in psg.lcut(content)]
    print(res)

获取分词结果中列表的top n
from collentions import Counter
seg_list = jieba.lcut("他来到繁华的魔都上海; 他曾想仗剑走天涯, 看尽繁华城市", cut_all = False)
print(f"返回列表: {seg_list}")
top2 = Counter(seg_list).most_commom(2)     # 从大到小排序返回第2个
print(top2)

orlsentence = "研究生命的起源吶"
基于正向最大匹配分词:
    从左到右匹配分词
    研究生-命-的-起源-吶

基于逆向最大匹配分词:
    从右到左匹配分词
    研究-生命-的-起源-吶
"""
class Onput(object):
    def __init__(self):
        self.window_size = 3    # 字典的最大次长度, 每次初次分词的滑动窗口
        self.dic = ['研究', '研究生', '生命', '起源']     # 加载词典库

    def cut(self, text):
        result = []   # 切分之后的结果
        index = 0     # 计数
        text_length = len(text)     # 统计text长度
        while text_length > index:
            for size in range(self.window_size+index, index, -1):      # 取(3, 0, -1) 逆向取, 三个字一检索,从第三个字开始往前找
                piece = text[index: size]       # size = 3  (0, 3) → (研究生) --在字典--> index = 2 --> break
                if piece in self.dic:
                    index = size - 1  # index = 2
                    break
            index = index + 1   # 第一轮出来为 index = 3  第二轮从3 开始 之后到 6
            result.append(piece)     # 不断检索, 直到有字典的内容否则在筛选后单个输出
        return result

text = '研究生命的起源吶'
tokenizer = Onput()
print(tokenizer.cut(text))          # 结果会依照字典里面的逆顺序优先选

# 字典的分词定义
"""
分词方式:
    1. 正向最大匹配
        从左到右匹配分词

    2. 逆向最大匹配
        从右到左匹配分词

    3. 双向最大匹配
        将正向最大匹配法得到分词结果和逆向分词匹配法结果进行比较, 选取最优
        分词数不同: 取分词量(单个字)较少的那个
        相同: 没有歧义

准确率评估:
    衡量验证集或者测试集预测的标签与真实值标签是否相等的指标:
    accuracy = 预测标签与真实标签相同的数量 / 总的预测数据集数量

衡量标准:
    1. 精准率
    2. 召回率
    3. F1

数据内容:
    1. train 标准答案(真值): 实际分词效果
    2. segment 分词结果(效果): 模型切分出的所有词语数
    3. test  
    关系:
        p(精确率) = 模型准确切分词语数(重合部分/交集) / 模型切分出的所有词语数(segment)
        R(召回率) = 模型准确切分的词语数(重合部分/交集) / 标准答案(train)
"""
import re
def to_region(segmentation):
    reg = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        reg.append((start, end))
        start = end
    return reg

import numpy as np
def prf(target, pred):
    A_size, B_size, A_cap_B_size = 0, 0, 0
    for g, p in zip(target, pred):
        A, B = set(to_region(g)), set(to_region(p))
        A_size += len(A)        # 真实答案
        B_size += len(B)        # 分词预测结果
        A_cap_B_size += len(A & B)     # 取A B 交集
    q, r = A_cap_B_size / B_size, A_cap_B_size / A_size     # 计算精准率和召回率
    f = 2 * q * r / (q + r)
    return q, r, f

target = ['我 爱 北京', '我 爱 南京']
pre = ['我 爱 北京', '我 爱南京']
print(prf(target, pre))

# flask 内容
import flask
from flask import request, jsonify
import jieba
import numpy as np

# 第一步, 创建一个app
app = flask.Flask(import_name=__name__)
# 创建模型对象
class classifmymoid(object):

    def __init__(self):
        self.classes = ['负向评论', '正向评论']
        self.default_class_name = self.classes[1]
        self.default_probility = 0.5

        # 恢复
        jieba.load_userdict('jieba_words.txt')
        word_dict: Dictionary = Dictionary.load('word_dict.pkl')
        print(f'单词数目: {len(self.word_dict)}')
        # net = torch.load('路径')
        self.net = Network(
            num_embeddings = len(self.word_dict) + 2,
            embedding_dim = 128,
            n_class = 2,
            hidden_size = 64,
            # embedding_table = np.pad(word2vec_embedding_table, ((2, 0), (0, 0)), constant_valuse = 0.0)
        )
        state_dict = torch.load('params.pt')
        self.net.load_state_dict(state_dict)    # 恢复
        self.net.eval()      # 将模型设置为预测模型

    def word_2_index(self, words, max_lenght=20):
        word = self.word_dict.doc2idx(words)    # 转换成序号(如果不存在返回-1)
        # 保证长度一致
        if len(words) < max_length:
            for i in range(max_length - len(words)):
                words.append(-2)    # 填充
        words = words[: max_length]    # 截断
        words = np.asanyarray(words) + 2
        return words

    def interface(self, text):
        """
        针对给定文本调用训练好的模型进行预测, 预测结果为: 预测类别, 预测概率值
        : param text: 待预测文本
        : return: 预测类别, 预测概率值  
        """
        if len(text) == 0:
            raise ValueError(f'模型输入不能为空, 当前输入参数为{text}')
            # 预测
        words = split_text(text)
        words = self.word_2_index(words)
        with torch.no_gard():
            _input = torch.from_numpy(np.asanyarray[words]).to(torch.long).T
            y_ = self.net(_input)
            print(y_)
            print(y_.shape)
            p_ = torch.softmax(y_, dim = 1).cpu().numpy()[0]
            y_ = y_.cpu().numpy()[0]        # 获取第一条样本的置信度
            y_ = np.agrmax(y_)      # 选择置信度最大的下标  
            print(f'文本{text}, 预测结果为{self.classes[y_]}--{p_}')
        return self.classes[y_], p_

model = classifmymoid()

# 第二部创建进入服务器方法
@app.route('/')   # 加注解, 指定资源
@app.route('/index')
@app.route('/f', methods=['GET', 'POST'])
def index():
    print("进入到服务器的方法")
    return "返回一个字符串"

# 第三步, 创建请求方式
@app.route('/t1', methods=['GET', 'POST'])   # 访问方式
def t1():
    _args_dict = None       # 字典
    _method = request.method
    if _method == 'GET':
        _args_dict = request.args   
    elif _method == 'POST':
        _args_dict = request.form     # 参数
        # 如果是post的文件上传参数, 需要通过request,files获取
    else:
        raise ValueError(f"不接受当前访问支持")
    # 获取name 的参数
    name = _args_dict['name']   # 必传参数
    age = _args_dict.get('age', -1)     # 可选参数
    print(f'请求方式:{_method}, 获取参数为{name}--{age}')
    print('基于传入的参数进行数据处理, 然后调用相关逻辑代码, 最终返回结果, 一般情况下用json字符串的形式返回')
    result = {
        'code': 200,
        'msg': '操作成功',
        'data': [{'name': name,
            'age': age,
            'sex': sex
            },
            {
                'name': f'{name}2',
                'age': age + 1,
                'sex': sex
            }
        ]
    }
    return jsonify(result)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
    # 获取参数
        _args_dict = request.args if request.method == 'GET' else request.form
        text = _args_dict.get('text', "").strip()
    # 基于参数进行相关逻辑代码执行
        if len(text) == 0:
            r = {
                'code': 202,
                'msg': f'请求数据异常, 请给定有效长度的text文本参数'
            }
        else:
            p_class, p_prob = model.interface(text)
            r = { 
                'code': 200,
                'msg': '执行成功',
                'data': {
                    'text': text,
                    'calssname': p_class,
                    'probability': float(p_prob)
                }
            }
    except Exception as e:
        # 异常捕获
        r = {
            'code': 201,
            'msg': f'服务器异常{e}',
            'data': {
                'calss_name': model.default_class_name,
                'probability':  float(model.default_probility)
            }
        }
    # 逻辑执行结果返回
    return jsonify(r)

if __name__ == "__main__":
    # 启动flask
    app.run(
        host = "0.0.0.0",   # 给定监听的ip地址, 0.0.0.0是监听所有ip地址
        port = 5000,    # 给定监听端口号
        debug = True    # 代码更改后是否自动更新
    )
    model = classifmymoid()
    while True:
        text = input('请输入你的评论: ')
        if text == ':':
            break
        else:
            print(model.interface(text))
    print(model.interface('挺辣的, 很好吃'))


# attention
"""
注意力机制, 多个时刻进行融合:
    value1 = t1(data)   特征值
    value2 = t2(data)
    value3 = t3(data)
    value4 = t4(data)   
提取特征信息(特征向量):
    计算相关性
    计算query(查询), 每个时刻key的相似度:
        query=Q
        keys=K
        similar1 = F(Q, K)1
        similar2 = F(Q, K)2
        similar3 = F(Q, K)3
        similar4 = F(Q, K)4
        或者
        similar1 = t1与t1相似度   Q为t1, k为t1
        similar2 = t1与t2相似度   Q为t1, k为t2
        similar3 = t1与t3相似度   Q为t1, k为t3
        similar4 = t1与t4相似度   Q为t1, k为t4
    相似度转为权重---softmax归一化
        weight: 
            a1=softmax(F(Q, K)1)
            a2=softmax(F(Q, K)2)
            a3=softmax(F(Q, K)3)
            a4=softmax(F(Q, K)4)
合并:
    attention_value1 = a1*value1+a1*value2+a1*value3+a1*value4
    attention_value2 = a2*value1+a2*value2+a2*value3+a2*value4
    attention_value3 = a3*value1+a3*value2+a3*value3+a3*value4
    attention_value4 = a4*value1+a4*value2+a4*value3+a4*value4
如何让不同时刻值不同:
    t1=f(q1, k1, v1)
    t2=f(q2, k2, v2)
    t3=f(q3, k3, v3)
    t4=f(q4, k4, v4)
    由于q1, q2, q3, q4不同, 因此每个时刻值不同
流程:
    Q(搜索词)--->K(文本, 样本, 商品)---->V(商品向量)
"""


"""
以下是NLP中常用算法的一些具体内容：

1. 分词算法

（1）最大匹配算法：从后往前扫描输入文本，并根据词典中最长的词语长度选择合适的分词。

（2）正向最大匹配算法：顺序扫描输入文本，根据词典中最长的词语长度选择合适的分词。

（3）逆向最大匹配算法：从后往前顺序扫描输入文本，根据词典中最长的词语长度选择合适的分词。

（4）双向最大匹配算法：同时对正向和反向进行扫描和匹配。

2. 词性标注算法

（1）隐马尔可夫模型(HMM)：基于概率理论，将分词结果与词性进行联合识别。

（2）最大熵模型(MaxEnt)：最大化条件熵，通过训练模型得到最大的概率或熵。

（3）条件随机场(CRF)：根据概率分布计算出最优的词性标注序列。

3. 命名实体识别算法

（1）规则匹配：基于预定义的识别规则进行匹配和识别。

（2）最大熵模型(MaxEnt)：基于训练数据集，使用最大熵估计方法来预测每个词的标签。

（3）条件随机场(CRF)：使用条件随机场模型对命名实体进行标注和识别。

4. 句法分析算法

（1）基于规则的方法：使用规则和语法进行分析，如语法树、依存关系树等。

（2）PCFG：使用上下文无关文法进行概率分析，提高语法分析准确度。

（3）CYK：使用运行时间较快的CYK（Cocke-Younger-Kasami）算法进行文法解析。

（4）图转移算法：使用依存图转移来建立句子的依存关系，如Arc-standard、Arc-hybrid等。

5. 语义分析算法

（1）基于词汇的方法：使用同义词、近义词等方式来分析和推断语义。

（2）向量化表示方法：使用词向量、段落向量、文档向量等方式进行表达和分析。

（3）推理模型：基于逻辑推理的方式来对语义进行分析和推断，如谓词逻辑、模态逻辑等。

6. 机器翻译算法

（1）统计机器翻译（SMT）：根据翻译语料库，使用统计模型进行机器翻译。

（2）神经机器翻译（NMT）：基于深度神经网络模型，学习输入端和输出端之间的非线性映射关系，进行机器翻译。

7. 情感分析算法

（1）基于词典的方法：使用预定义的情感词典进行情感分析。

（2）基于机器学习的方法：使用分类器，如朴素贝叶斯、神经网络分类器等进行情感分析。

（3）基于深度学习的方法：如使用LSTM、GRU等模型进行情感分析，提高情感识别的准确度。
"""

"""
以下是一些常用的NLP多媒体处理方法：

1. 文本分类：根据给定文本的内容将其分类到一个或多个预定义类别中，比如垃圾邮件过滤、情感分析、新闻分类等。常用算法：朴素贝叶斯、支持向量机、决策树、深度学习（如卷积神经网络、循环神经网络等）。

2. 命名实体识别：从文本中识别出特定类型的命名实体，比如人名、地名、机构名、时间、日期等。常用算法：条件随机场、支持向量机、深度学习（如循环神经网络、门控循环单元等）。

3. 关系提取：从文本中提取出实体之间的语义关系，包括事件、事实、属性等。常用算法：基于规则的方法、远程监督、半监督学习、深度学习（如卷积神经网络、注意力机制等）。

4. 文本生成：根据给定的上下文和任务，自动生成与之相应的文本。常用算法：基于规则的方法、神经网络语言模型、生成对抗网络（GAN）、变分自编码器（VAE）等。

5. 文本摘要：从长文本中提取出关键信息，生成一个简洁的摘要。常用算法：基于统计的方法、基于图像的方法、深度学习（如长短时记忆网络、注意力机制等）。

6. 文本翻译：将一种语言的文本转换成另一种语言的文本。常用算法：统计机器翻译、神经机器翻译、深度学习（如基于注意力机制的翻译模型等）。

7. 图像分类：将图像分类到不同的类别中，比如动物、汽车、建筑等。常用算法：卷积神经网络、支持向量机、决策树等。

8. 目标检测：从图像中检测出特定的物体。常用算法：RCNN、Fast RCNN、Faster RCNN、YOLO、SSD等。

9. 图像分割：将图像中的不同区域分割开来，并对每个区域进行分类或赋予其特定的语义信息。常用算法：基于阈值的方法、基于区域的方法、基于边缘的方法、深度学习（如U-Net、Mask R-CNN等）。

以上是常用的NLP多媒体处理方法，它们都有各自的优点和适用场景，需要针对具体问题进行选择。
"""