# 流程架构
"""
物料库(百万级)--百万级-->
召回(规则召回, 协同召回, 向量召回)--万千-->
过滤(黑名单过滤, 兴趣, 白名单过滤)--万千-->
粗排(少量特征, 简单模型, 机器学习/深度学习内容)--千百-->
精排(全特征复杂模型)--百十-->
重排(规则排序)

数据采集(用户数据, 行为数据, 内容数据)--> 数据处理(清洗, 归并, 标注) --> 画像体系(用户画像, 业务画像) --> 算法引擎(算法模型) --> 推荐接口(相关推荐, 个性化推荐, 热门推荐)
"""

# 推荐系统评估
"""
召回率(Recall): 在推荐列表中, 用户实际点击商品和用户所有点击商品之间占比
精确率(Precision): 在推荐列表中, 用户实际点击的商品占比
详情参见混淆矩阵
Pt 为阈值, 当P (置信度) > Pt 时, 不采用, 通不过检验

ROC_AUC 曲线:
    ROC:
        使用正假率(FPR)做为横坐标, 正例率(TPR)作为纵坐标, 是分类算法模型评估方式, 是一个非递减曲线
    AUC:
        形成原因: ROC曲线不太直观
        作用: AUC越大表示模型的区分能力越强效果越好

MAP(Mean Average Precision)是对精准率的一个修正:
    AP@u = ∑(precision@N) * rel(N) / |R^|   @N 取前N个用户
    MAP = Σ(AP@u)/abs(U)  @u 针对用户u

NDCG: 类似MAP的一种考虑序列位置的一种推荐, 搜索领域评估指标
    CG@k = ∑ rel_i
    DCG@k = ∑ ((2**rel_i - 1) / log2(i+1))
    NDCG@k =  DCG@k / IDCG@k   ---> IDCG@k 取最大值 -> 变成概率
    IDCG@k 按照实际点击做排序, 按照最优(rel_i)的值排序
"""

# 排序模型
"""
方法:
    1. 单点法
        不考虑相关性, 直接转为分类/回归问题, 直接使用常规机器学习, 深度学习算法建模
        复杂度低
        标注简单, 可直接会用用户的点击反馈等信息
        缺点: 没有考虑相关性, 导致最终模型无法学习商品与商品之间顺序
    2. 配对法
        为了解决单点法无法对序列内商品之间关系建模问题, 基本思路是对样本两两比较, 构建偏序商品
        进行完美排序
        缺点: 1.完美排序不存在 2.序列长度不一
        常用算法:
            BPR贝叶斯个性推荐
            常用于精排模块
    3. 列表法
        直接优化 NDCG这样指标, 从而学到最佳结果
区别:
    损失函数, 标注方式和优化方法不同

分类:
CTR模型:
    点击率预估模型, CTR模型---> 单样本(1会, 0不会)
    未考虑商品相关性
"""

# suprise算法框架
"""
算法:
    random_pred.NormalPredict   # 随机预测打分, 假定用户的打分分布服从高斯分布
    baseline_only.BaselineOnly  # 基于统计的就基准线预测打分
    knns.KNNbasic               # 基本协同过滤算法
    knns.KNNWithMeans           # 基本协同过滤算法变种, 考虑每个用户平均评分
    knns.KNNWithZScore          # 基本协同过滤算法变种, 考虑每个用户评分归一化操作
    knns.KNNBaseline            # 基本协同过滤算法宾中, 考虑基于统计基准线评分
    matrix_factorization.SVD    # SVD矩阵分解算法
    matrix_factorization.SVDpp  # SVD++矩阵分解算法
    matrix_factorization.NMF    # 基于非负矩阵分解协同过滤算法
    slope_one.SlopeOne          # SlopeOne协同过滤算法
    co_clustering.CoClustering  # 一种基于协同聚类的协同过滤算法
相似度度量标准:
    cosine: 计算所有用户或者所有物品之间的余弦相似度
    msd: 计算所有用户或者所有物品之间的平均平方差相似度
    person: 计算所有用或者所有物品直接person(卡方检验)相似度
    person_baseline: 使用基准线方式计算person相似度
评估准则:
    RMSE(回归系统的拟合标准差): 均方根误差
    MSE: 预测数据和原始数据对应点误差的平方和的均值
    MAE: 平均绝对误差, 又被称为 L1范数损失
    FCPS: 一致序列对比率评分，计算评分一致的物品对在所有物品对中的占比
"""

# Normal Predict
"""
核心思想: 认为用户对于物品评分是服从高斯分布的一个随机过程
假定用户对物品的评分数据是服从正态分布, 从而可以基于正态分布的期望μ和标准差δ随机的给出当前用户对于其他物品评分
基于大数定律和中心极限定律或者使用最大似然估计(MLE)我们可以得到每个用户评分数据所属正态分布对应期望μ和标准差δ
μ = 1 / |R_train| * ∑ r_ui
δ**2 = 1 / |R_train| * ∑ (r_ui - μ**2

"""
import os
import surprise
from surprise import accuracy
from surprise import Dataset, Reader
from surprise import NormalPredictor, BaselineOnly
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
# 基本用法
'数据加载'
"方法1"
data = Dataset.load_builtin('ml-100k')    # 加载内部数据集---内容放置用户根目录
"方法2"
# 指定文件所在路径(要求给定的文件中只有数据, 没有列)
file_path = os.path.expanduser('./datas/ratings.csv')
# 必须给定数据格式()(必须给定一个数据读取器, 告诉如何读取数据)
reader = Reader(line_format='uesr iiitem rating timestamp', sep=',')
# 加载数据
data =  Dataset.load_from_file(file_path, reader=reader)

'数据划分'
trainset, testset = train_test_split(data, test_size=0.25)

'模型构建'
algo = NormalPredictor()
cross_validate(algo, data, mmeasures=['RMSE', 'MAE', 'fcp'], cv=5,
                return_train_measures=True, verbose=True)
"""注: 进行模型训练的时候, 必须对训练数据进行构建(必须构建)----->将Movieline格式数据转化为稀疏矩阵, 在这个转换过程, 会做一个id转换, 就是将外部数据id转化内部数据id"""
trainset = data.build_full_trainset()
# 模型训练
algo.fit(trainset)

'预测----> 用户id<->商品id'
y_ = algo.predict('用户id', '商品id', '评分')
print(f'y预测评分:{y_.est}')

'评估'
"方式1"
predictions = algo.test(testset)
print(f'RMSE:{accuracy.rmse(predictions)}')   # 均方根误差
print(f'MSE:{accuracy.mse(predictions)}')    # 均方差
print(f'FCP:{accuracy.fcp(predictions)}')    # 一致序列对比率评分
"方式2"
cross_validate(algo, data, measures=['RMSE', 'MSE', 'fcp'], cv=5, 
                return_train_measures=True, verbose=True)

# Baseline Only
"""
Baseline基线: 
    也称为基准线---平均值那条线
    相对于平均值的偏差, 基于统计数据的预测算法
    原理:
        认为每个用户对于每个商品评分是由三部分钩成:
            1. 评分的均值μ
            2. 用户的评分基线b_μ
            3. 当前物品的评分基线b_i
    公式:
        r_^(预测) = b_ui = μ + b_u + b_i
        J(b_u, b_i) = Σ(r_ui-μ-b_u-b_i)**2 + λ(Σb_u**2+Σb_i**2)
        b_u, b_i = minJ(b_u, b_i)
"""
'模型参数-模型对象构建'
_bsl_options = {
        'method': 'als',    # 指定计算方式, 默认als, 可选sgd
        'n_epoch': 10,      # 迭代次数
        'reg_i': 25,        # b_i 计算过程中的正则化项
        'reg_u': 10,        # b_u 计算过程中的正则化项
        'learning_rate': 0.01  # 学习率
}   
bsl_options = {
    'method': 'als',    # 指定计算方式, 默认als, 可选sgd
    'n_epoch': 10,      # 迭代次数
    'reg': 0.02,        # 正则化系数
    'learning_rate': 0.01  # 学习率
}

'模型加载'
algo = BaselineOnly(bsl_options=_bsl_options)

'模型训练'
cross_validate(algo, data, measures=['RMSE', 'MSE', 'fcp'], cv=5,
                return_train_measures=True, verbose=True)
                
'预测'
y_ = algo.predict('110', '120', 3.0)
print(f'预测评分:{y_.est}')
y_ = algo.predict('greey', '120', 3.0)
print(f'预测评分:{y_.est}')  # 加入.est表示截断

'评估'
predictions = algo.test(testset)
print(f"fcp: {accuracy.fcp(predictions)}")
# 协同过滤
"""
协同过滤(CF, Collaboratiive Filtering)也叫基于近邻的推荐算法
思路: 
    找近邻, 找相似, 利用已有的用户群过去的行为或者意见来预测数据, 根据相似推荐结果, 类似knn
算法输入:
    用户-物品评分矩阵
    矩阵类似于excel表格结构:
        用户    物品    评分    时间标签
主要方式:
    基于用户的最近邻推荐UserCF
    基于物品的最近邻推荐ItemCF
"""

# 协同过滤_UserCF
"""
主要思想:
    1. 对输入的评分数据集和当前用户ID作为输入
    2. 找出与当前用户过去有相似偏好的其他用户
    ---> 这些用户叫做对等用户或者最近邻
    3. 对当前用户没有见过的每个产品p, 利用用户近邻对产品p的评分进行预测
    4. 选择所有产品评分最高的前几个产品推荐给当前用户
前提:
    用户喜欢物品相似度
    用户偏好不随时间变化
流程:
    1. 计算所有用户相似矩阵(基于共同评论商品列表)
        - 提取评分矩阵中用户共同评分的商品评分
        - 计算两个向量的相似度直接作为用户相似度
            a. 余弦相似度
                sim = cos(Θ) = A * B / (|A| * |B|) = Σ(Ai * Bi) / (Σ(Ai)**2)**(1/2) * (Σ(Bi)**2)**(1/2)
            b. 欧几里得距离
            c. 皮尔森卡方检验
        - 重复以上两个操作, 计算所有商品评分
    2. 获取物品评分
        - 提取和某用户评分相似的k个用户
        - 基于相似矩阵发现最相似的是U1
        - U1 -> p1 -> 5
        - 将所有相似用户的商品p1上的评分合并
            均值合并: 
                r_^ui = ∑sim(u, v) * r_vi / Σ sim(u, v) + b_ui 
                用户u的k个近邻, 在商品v上有评分, 之后加权, b_ui: 偏移(需要学习)
            (r1 + r2 + ---- + rk) / k
            k==1 --> 预测评分等于5

        获取用户u最相似k近邻
        根据k近邻用户对物品i评分计算当前对物品i的评分
    3. 重复2 计算当前用户的所有物品评分
    4. 重复2, 3 计算所有用户的所以物品评分
    5. 提取评分排序, 评分最高的N个商品作为推荐商品
"""
# 计算用户与用户之间的相似矩阵
"""底层构建"""
import surprise
from surprise import KNNBaseline
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import Dataset
'构建模型'
sim_options = {'name': 'pearson', 'user_based': True}
algo = KNNBaseline(sim_options=sim_options)

'模型训练'
cross_validate(algo, data, measures=['RMSE', 'MSE', 'fcp'], cv=5,
                return_train_measures=True, verbose=True)

'模型参数的给定'
sim_options  = {
    'name': 'pearson',      # 使用什么相似度度量方式
    'user_based': True      # 是UserCF还是ItemCF---参数为true时, 表示UserCF
}

'模型构建'
algo = KNNBasic(sim_options=sim_options)
# 进行模型训练时候, 必须对训练数据进行构建---在这个过程中, 会做一个onehot的id转换
trainset = data.build_full_trainset()
'模型训练'
algo.fit(trainset)

'计算推荐列表'
row_all_user_ids = ['1', '3', '5', '7', '9']    # 所有的用户id
row_all_item_ids = ['2', '4', '6', '8', '10']   # 所有的物品id
for user_id in row_all_user_ids:
    for item_id in row_all_item_ids:
        print(f'用户{user_id}对于物品{item_id}评分为: {algo.predict(user_id, item_id)}')

'获取用户u对物品i的评分'
row_user_id = '1'   # 对应文件中的用户id
row_item_id = '10'    # 对应文件中的商品i
inner_user_id = algo.trainset.to_inner_uid(ruid=row_all_user_ids)    # 获取内部id转换后的用户id
inner_item_id = algo.trainset.to_inner_iid(ruid=row_all_item_ids)

# 获取近邻
row_user_id = '1'
inner_user_id = algo.trainset.to_inner_uid(ruid=row_user_id)    # 获取内部id后转换为用户id
user_neigbors = algo.get_neighbors(inner_user_id, k=10)  # 基于内部用户id计算最近邻
print(user_neigbors)

"--实际流程"
import surprise
from surprise import KNNBaseline
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy

'数据划分'
trainset, testset = train_test_split(data, test_size=.25)
'构建模型'
_sim_options = {'name': 'pearson', 
               'user_based': True,   # UserCF算法 
            #    'user_based': False,    # ItemCF算法
               'min_support': 1
}

'模型构建'
algo = KNNBasic(k=40, min_k=1, sim_options=_sim_options)

'训练'
algo.fit(trainset)

'预测'
y_ = algo.predict('110', '120', 3.0)   # predict(用户id, 商品id, 评分)
print(f"预测评分: {y_.est}")
y_ = algo.predict('110', '120', 4.0)
print(f'预测评分: {y_.est}')

'评估'
predictions = algo.test(testset)
print(f"fcp: {accuracy.fcp(predictions)}")

# 协同过滤_ItemCF
"""
思想: 
    基于物品的最近邻推荐----> 物品间的相似度, 而不是基于用户之间的相似度来进行预测评分
与UserCF比较区别:
    UserCF: 计算用户之间相似度, 将用户喜好物品推荐给当前用户
    ItemCF: 计算物品之间相似度, 根据当前用户喜好物品来推荐其他物品列表
前提/假设:
    相似度高推荐给用户
    用户的偏好不随时间改变
流程:
    1. 计算商品与商品的相似矩阵(基于两个商品同时被用户评论的共同用户列表)
    2. 用户u对当前物品的评分方式如下:
        -- 获取当前物品i最相似的k个近邻物品(这几个物品被u评论过)
        -- 用户u对k个近邻物品的评分计算当前用户i的评分
    3. 重复2, 计算当前用户对所有物品评分
    4. 重复2, 3 计算所有用户对所有物品评分
    5. 对于某个用户, 提取该用户对所有物品的评分排序后, 评分最高N个商品作为推荐商品列表
合并:
    首先获取出k个最相似的近邻物品, 然后将当前用户在这些物品上的评分加权求和

"""
"""底层构建"""
import surprise
from surprise import KNNBaseline
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import Dataset
'构建模型'
sim_options = {'name': 'pearson', 'user_based': True}
algo = KNNBaseline(sim_options=sim_options)

'模型训练'
cross_validate(algo, data, measures=['RMSE', 'MSE', 'fcp'], cv=5,
                return_train_measures=True, verbose=True)

'模型参数的给定'
sim_options  = {
    'name': 'pearson',      # 使用什么相似度度量方式
    'user_based': True      # 是UserCF还是ItemCF---参数为true时, 表示UserCF
}

'模型构建'
algo = KNNBasic(sim_options=sim_options)
# 进行模型训练时候, 必须对训练数据进行构建---在这个过程中, 会做一个onehot的id转换
trainset = data.build_full_trainset()
'模型训练'
algo.fit(trainset)

'计算推荐列表'
row_all_user_ids = ['1', '3', '5', '7', '9']    # 所有的用户id
row_all_item_ids = ['2', '4', '6', '8', '10']   # 所有的物品id
for user_id in row_all_user_ids:
    for item_id in row_all_item_ids:
        print(f'用户{user_id}对于物品{item_id}评分为: {algo.predict(user_id, item_id)}')


'直接获取相似的物品'
row_item_id = '1'    # 对应文件中物品id
inner_item_id = algo.trainset.to_inner_iid(riid=row_item_id)
item_neiighbors = algo.get_neighbors(inner_item_id, k=10)
'获取这些相似用户的这个相似度'
for item in item_neiighbors:
    row_like_item_id = algo.trainset.to_raw_iid(item)
    print(f'物品{row_item_id}和物品{row_like_item_id}的相似度为{algo.sim[inner_item_id][item]}')

'获取近邻'
row_user_id = '1'
inner_user_id = algo.trainset.to_inner_uid(ruid=row_user_id)    # 获取内部id后转换为用户id
user_neigbors = algo.get_neighbors(inner_user_id, k=10)  # 基于内部用户id计算最近邻
print(user_neigbors)

'相似度矩阵输出'
print('物品相似度矩阵: ')
print(algo.sim)
pro_2_pro_sim_path = '物品相似度数据路径'
with open(pro_2_pro_sim_path, 'w', encoding='utf-8') as writer:
    n = algo.sim.shape[0]
    for i in range(n):
        # 针对物品i不保存所有物品及它的相似度--> 提取相似的k个物品
        for j in range(n):
            # 转换成为外部实际物品id
            row_i = algo.trainset.to_raw_iid(i)
            row_j = algo.trainset.to_raw_iid(j)
            # 获取两个物品直接相似度
            sim = algo.sim[i][j]
            # 输出
            writer.writelines(f"{row_i}, {row_j}, {sim}")

"--实际流程"
import surprise
from surprise import KNNBaseline
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy

'数据划分'
trainset, testset = train_test_split(data, test_size=.25)
'构建模型'
_sim_options = {'name': 'pearson', 
            #    'user_based': True,   # UserCF算法 
               'user_based': False,    # ItemCF算法
               'min_support': 1
}

'模型构建'
algo = KNNBasic(k=40, min_k=1, sim_options=_sim_options)

'训练'
algo.fit(trainset)

'预测'
y_ = algo.predict('110', '120', 3.0)   # predict(用户id, 商品id, 评分)
print(f"预测评分: {y_.est}")
y_ = algo.predict('110', '120', 4.0)
print(f'预测评分: {y_.est}')

'评估'
predictions = algo.test(testset)
print(f"fcp: {accuracy.fcp(predictions)}")

# 协同过滤优缺点
"""
忧点:
    简单性: 实现简单, 而且在调整参数过程中只有一个邻近数需要调整
    合理性: 对于预测推荐提供了简洁并直观的理由
    高效性: 基于临近方法的推荐的效果特别高, 因为可以先进行预处理, 构建出相似矩阵, 实际中--->提供近似实时推荐结果
    稳定性: 当相似度矩阵构建完成后, 如果有用户对物品产生新的评分, 那么影响范围很小
缺点:
    显式评分: 通过问卷方式收集商品评分, 忧点准确, 缺点不评
    隐式评分: 点击即显式正向, 可以转为评分值, 优点数据多, 缺点评分可能不准
    覆盖有限: 两个用户之间相似性基于物品的评分, 而且只有相同物品评分才能作为近邻, 所以有局限性
    稀疏数据敏感: 用户只会评分一部分物品, 造成评分稀疏性, 导致两个用户或者物品之间的相似性计算仅适用少量有限近邻
    冷启动: 相似性权重计算可能依赖小部分评价, 导致推荐偏差---> 冷启动
冷启动问题:
    定义:
        新用户没有评论任何商品, 如何产生推荐? 如何将新商品推送给用户?
    解决:
        1. 利用混合方法进行推荐, 采用多种机器学习/深度学习模型完成推荐
        2. 采用ItemCF算法中的相似度矩阵策略缓解冷启动-->看一个商品, 把相似商品推荐
        3. 利用新品推荐强制推送
        4. 利用热门推荐给新用户推送商品列表
"""

# SVD矩阵分解
"""
隐语义模型---潜在因子算法:
    思路: 
        用户对物品高评分是物品中所包含偏好信息恰好就是用户喜欢的信息
        偏好无法简单明显找出, 认为这个偏好就是用户对物品评价的因子
        得---用户<->潜在因子矩阵Q和物品<->潜在因子矩阵P可以计算出物品评分
            R_^ = Q*P.T
        假定有n个用户, m个物品, k个因子, R为用户-物品评分矩阵, Q为用户-潜在因子矩阵, P为物品-潜在因子矩阵
SVD矩阵分解:
    根据已有评分, 分析评分者对各个物品因子的喜好程度, 以及物品包含这些因子, 最后反过来预测评分
    通过SVD的方式可以找出影响评分的显示因子和隐藏因子   
    数学定义:
        给定评分矩阵R分解为三个矩阵的乘积, 其中U V称为左(U), 右(V)奇异向量
        Σ对角线上的值称为奇异值, 其中R 为n*m的矩阵, U为n*n矩阵, Σ为n*m矩阵, V为m*m矩阵
        可以使用前k个奇异值来近似代替R矩阵
        公式:
            R_(n*m) = U_(n*n) * Σ_(n*m) * V_(m*m)
            R_(n*m) ≈ U_(n*k) * Σ_(k*k) * V_(k*m).T
条件:
    数学方式--SVD分解要求矩阵稠密, 不能有空白, 如果矩阵稠密(已经找到所有用户物品的评分)不需要SVD
目的:
    找出各类物品因子-用户评分, 使矩阵变稠密
"""

# SVD矩阵分解 FunkSVD
"""
将矩阵分解为两个矩阵来降低执行消耗, 将U V矩阵进行转换得到用户因子矩阵Q和物品因子矩阵
公式:
    R_ui ≈ Q_(n*k) * P_(m*k)
    Q_(n*k) = U_(n*k) * (Σ_(k*k))**(1/2)
    P_(m*k) = (Σ_(k*k))**(1/2) * V_(m*k).T).T
    r_^ui = q_u * p_i.T = Σ(q_uk*p_ik)

用户因子矩阵Q和物品因子矩阵P的计算可以利用线性回归的思想:   
    算法:
        1. 通过随机梯度(SGD)进行学习
        2. 迭代式更新相关参数
    特点: 
        SVD矩阵因子分解推荐算法对稀疏矩阵也可处理, 不用计算误差
    公式--预测r_^ui:
        // α 是学习率
        r_^ui = q_u * p_i.T
        e_ui = r_ui - r_^ui
        min(1/2 * Σ(r_ui - q_u*p_i.T)**2
        p^(k+1)_i = p^k_i + α*e_ui*q^k_u
        q^(k+1)_u = q^k_u + α*e_ui*p^k_i    // k+1表示第几轮
    如果过拟合---加入正则:
        r_^ui = q_u * p_i.T
        e_ui = r_ui - r_^ui
        min(1/2 * (Σ(r_ui - q_u*p_i.T)**2 + λ(|q_u|**2+|p_i|**2))
        p^(k+1)_i = p^k_i + α*(e_ui*q^k_u - λ*p^k_i)
        q^(k+1)_u = q^k_u + α*(e_ui*p^k_i - λ*q^k_u)  
"""

# 矩阵分解 BiasSVD
"""
更改FunkSVD预测值供公式, 让基准评分/偏置项基础上引起最终预测值的变化
公式:
    r_^ui = μ + b_u + b_i + q_u*p_i.T
    e_ui = r_ui - r_^ui
    // 加入正则
    min(1/2 * (Σ(r_ui - r_^ui)**2 + λ((b_u)**2 + (b_i)**2 + |q_u|**2+|p_i|**2))
    // 梯度下降(BGD)更新:
    e_ui = r_ui - r^_ui
    // α 是学习率
    b^(k+1)_u = b^k_u + α*(e_ui - λ*b^_u)
    b^(k+1)_i = b^k_i + α*(e_ui - λ*b^_i)
    p^(k+1)_i = p^k_i + α*(e_ui*q^k_u - λ*p^_i)
    q^(k+1)_i = q^k_u + α*(e_ui*p^k_i - λ*q^_u)
"""

# 矩阵分解 SVD++
"""
在BiasSVD基础上加入用户隐式反馈---浏览行为, 点击行为, 每一个商品携带一个隐式反馈y_j
公式:
    r_^ui = μ + b_u + b_i + (q_u + |I_u|**(-1/2)*Σ(y_j))*p_i.T
"""
"--实现代码"
import surprise
from surprise import Dataset, Reader
from surprise import SVD            # pip install matrix_factorization 
from surprise.model_selection import cross_validate
from surprise import accuracy
'''
参数:
    1. n_factors: 隐因子数目, 默认100
    2. n_epochs: BGD迭代次数, 默认20
    3. biased: 给定模型中是否使用偏置项, 默认为True, 即BiasSVD分解模型. False, 表示用FunkSVD
    4. lr_all: 给定学习率, 默认0.005
    5. reg_all: 给定正则化系数, 全部参数使用相同正则化系数, 默认为0.02
'''
'加载数据'
data = Dataset.load_builtin('m1-100k')     # 加载内部数据集---内容放置用户根目录
'将数据转化为Dataset训练集'
dataset = data.build_full_trainset()
'模型对象构建'
algo = SVD(n_factors=10, n_epochs=10, reg_all=0.2, lr_all=0.05)
'训练'
algo.fit(dataset)
'预测'
algo.predict('用户id', '物品id')

"--实际代码"
data = Dataset.load_builtin('ml-100k')
'数据划分'
trainset, testset = train_test_split(data, test_size=.25)
'模型构建'
algo = SVD(n_factors=8, n_epochs=10)
'训练'
algo.fit(trainset)
'预测'
y_ = algo.predict('32', '4', 3.0)
print(f'预测评分: {y_.est}')
y_ = algo.predict('66', '2', 3.0)
print(f'未知用户评分: {y_.est}')
'评估'
predictions = algo.test(testset)
print(f'MSE: {accuracy.mse(predictions)}')    # 均方差
print(f'RMSE: {accuracy.rmse(predictions)}')    # 均方误差
print(f'fcp: {accuracy.fcp(predictions)}')    # 一致序列对比率评分

# 网格调参
from surprise.model_selection import GridSearchCV
data = Dataset.load_builtin('m1-100k') 
param_grid = {
    'n_epochs': [10, 20],    # 迭代次数
    'lr_all': [0.1, 0.5],    # 学习率
    'reg_all': [0.1, 0.5]    # 正则化系数
}

'定义网格搜索对象'
grid_search = GridSearchCV(SVD, param_grid=param_grid, measures=['RMSE', 'MSE', 'FCP'], cv=3)
'加载数据/数据进行K-Flod'
grid_search.fit(data)
print('='*100)
'输出调优参数列表--输出最优RMSE值'
print(grid_search.best_score['rmse'])
print(grid_search.best_params['rmse'])

# 关联规则
"""
数据挖掘:
Apriori算法:
    关联规则衡量指标: 支持度和可信度
    本质: 找出购物数据集中最频繁的K项集
    目的: 找出数据中频繁出现的数据集合, 同时认为这些频繁出现的数据集合中的数据项存在关联(相似性)
    作用: 优化网站中商品的排列位置, 将相似的物品推荐给正在浏览对应物品的客户, 增加经济收益-节约成本
    基本概念:
        交易集: 包含所有数据的一个数据集合, 每一条都为交易数据
        项: 交易集中的每个商品成为一个项
        模式/项集: 项的组合
        支持度: 一个项集在整个交易集中出现的次数/出现的频度
            ex: Suppport({A, C}) = 2 表示A和C同时出现的次数是2次
        最小支持度: 交易次数支持度达到最小支持度情况下, 该项才会被计算
        频繁项集: 项集支持度>=最小支持度
        置信度: 关联规则左件和右件同时出现的频繁程度, 值越大, 出现概率越高
        关联规则: LHS->RHS(confidence)--->客户购买了左件(LHS), 也可能买右件(RHS), 购买置信度为confidence
    公式:
        Support_rule(X, Y) = (X∩Y)的数据量 / 总数据量
        Confidence_rule(X, Y) = (X∩Y)的数据量 / 含X的数据量
    思路: 
        包含几个数不能重复
        需要迭代: 
        先搜索出一项集对应的支持度, 筛选剪枝去掉最小支持度的1项集, 然后对剩下的进行连接
        得到候选的频繁2项集, 筛选剪枝去掉最小支持度的2项集, 以此类推
    流程:
        输入: 数据集合, 阈值   输出: 最大的频繁k项集
        1. 扫描数据集, 得到所有出现过的1项集
        2. 令k=1
        3. 挖掘频繁k项集
            -- 扫描数据计算候选频繁k项集的支持度
            -- 筛选剪枝去掉最小支持度项集, 如果得到项集为空, 返回频繁k-1项集
            -- 重复直到项集只有一项, 则返回频繁k项集

FP Tree算法:
    只用扫描两次数据集, 利用树结构提高算法执行效率
    本质: 需要将所有数据缓存下来
    基础知识:
        1. 项头表: 记录所有的1项出现次数, 按照次数降序排列
        2. FP Tree: 将原始数据集映射到内存中的一棵FP Tree
        3. 节点链表: 在项头表中存的嘤链表, 链表存储具体的FP Tree中存储对应节点位置
    流程:
        1. 项头表和FP Tree的构建(同时构建节点链表)
        2. FP Tree的挖掘
        项头表构建:
            扫描所有数据
            得到所有1项集支持度
            删除低于阈值的支持度-->得到频繁1项集
            频繁1项集排序按照支持度降序排列, 放入项头表中

        FP Tree构建:
            1. 扫描数据, 删除每条数据非频繁1项集, 并按照支持度降序排列, 得到排序后数据集
            2. 将排序好的数据直接插入到FP Tree, 靠前为先祖节点. 靠后为子孙节点
                共用先祖节点: 在对应的共用先祖节点计数+1
                插入后, 没有那个数据节点, 会创建新节点, 则项头表对应的节点会通过节点列表连接上新节点, 直到所有数据插入FP树后, FP树建立完成
        频繁项集挖掘:
            3. 从项头表由下而上挖掘, 从条件模式基递归挖掘得到项头表项的频繁项集
                对自下而上的线上数据做组合
            4. 不限制频繁项集的项5数, 返回上一步的所有频繁项集, 否则只返回满足项数要取的频繁项集
关联规则应用:
    看了又看, 买了又买:
        1. 训练/挖掘:
            - 将一个会话中用户的所有浏览商品作为一个商品列表(交易列表)
            - 利用关联规则的算法挖掘频繁1项集和频繁2项集
            - 计算所有商品X-> Y的置信度
                pids = [1, 2, 3, 4, 5, 6, 7]
                for pidi in pids:
                    s1 = XX # 获取pidi单独出现的支持度
                    if s1 is None:
                        continue
                for pidj in pids:
                    if pidi in pidj:
                        continue
                    s2 = XXXX # 获取pidi和pidj同时出现的支持度
                    if s2 is None:
                        continue
                    c = s2 / s1 # 置信度
                    # 将pidi pidj c 保存至数据库/文件
                    # 或者处理后保存: 只包括pidi对应的置信度最高的前100个pidj

        2. 推挤/线上应用:
            - 当用户浏览当前商品的时候, 直接基于当前商品id从数据库中获取置信度最高的前N个作为推荐
            - 召回
"""

# 推荐算法 LR
"""
LR: 逻辑回归算法---> 线性分类 ---> 评判用户是否点击/购买
    找个线性分割线
    p = sigmoid(wx+b)
    loss = -Σ(y*lnp+(1-y)*ln(1-p))    # 交叉熵损失
    w_^ = w - α*(σloss / σw)
    b_^ = b - α*(σloss / σb)
    需要CTR(点击率)预估和CVR(转化率)预估, 也可将推荐转换为二分类
    CTR预估:
        用户点击为正例, 没点为负例
        曝光: 展示给用户商品

GBDT:
    基于树结构的集成算法来激进型LR模型前的特征组合或者特征转换/映射
    落在同一个叶子节点满足相同特征信息
流程: 
    先进行GBDT---特征处理
    再进行onehot---转换为稀疏矩阵
    之后进行LR---预测评估
"""

# 推荐算法 FM
"""
Poly2:
    将多项式扩展的特征工程操作直接嵌入到模型中
    问题: 当两个特征组合项(x_i和x_j)非0组合特征才有意义, 单独出现多, 组合出现少
    处理: 把w_ij 转化为向量
    公式:
        y(x) = w_0 + Σ(w_i*x_i) + Σ_i(1, n)Σ_j(1, n)(w_ij*xi*xj)
FM:
    目的: 解决稀疏数据训练效果差的问题, 降低时间复杂度
    将Poly2中组合特征权重转化为两个向量内积--->每个特征属性都训练其对应额特征向量/隐特征向量
    作用: 解决Poly2算法特征稀疏导致训练完成后效果差的问题
    学习训练: 每个特征都有的隐特征向量, 学习特征属性隐向量, 出现就学
    优缺点:
        优:
        1. FM隐向量就加入, 提高模型泛化能力, 不会因为使数据稀疏导致隐向量训练不充分
        2. FM时间复杂度不高, 训练和推理均可达到O(kn)级别
        3. 参数量减少
        缺:
        1. 特征和不同类型特征组合时只能用同一组特征隐向量
        2. 解释性不强
    公式:
        y(x) = w_0 + Σ(w_i*x_i) + Σ_i(1, n)Σ_j(1, n)([V_i·V_j]*xi*xj)  
        Σ(w_i*x_i)复杂度O(n)  Σ_i(1, n)Σ_j(1, n)([V_i·V_j]*xi*xj)复杂度O(kn**2)
    更新参数:
        BGD更新-->
                     | 1     if Θ is w_0      
            σy / σΘ =| x_i   if Θ is w_i 
                     | x_i * Σ(v_jf*x_j)-x_i**2 * v_if   if Θ is v_if 
        时间复杂度O(kn)

FM_向量召回
    任意弄一个数据/基于sklearn 随机一个二分类数据基于pytorch框架实现logisitic回归模型
    在线执行:
    user log in ---> user Embedding ---> 快速存取--->检索(余弦距离)--->[item1, item2, item3]
    离线执行:
    user Feature ---|F 模|---user Embedding---数据库
    Item Feature ---|M 型|---item Embedding---数据库
"""             

# 推荐算法 FFM
"""
在FM基础上引入域的概念, 相当于将n个特征属性划分为f个域, 每个特征每个域学习一个隐向量
使模型表达能力更强, 复杂度会变高, 所以FMM复杂度为O(kn**2)
公式:
    y(x) = w_0 + Σ(w_i*x_i) + Σ_i(1, n)Σ_j(1, n)([V_if_j·V_jf_i]*xi*xj)      
    V_if_j  i 为这个特征 f_j为对应特征在哪个域
    0号域  x1  x2  x3
    1号域  x4  x5  x6
    2号域  x7  x8  x9  x10
    <v_20, v_30>  x2, x3
    <v_42, v_81>  x4, x8
问题:
    LR, FM, FFM输入特征为了离散化类别特征(OneHot处理), 如果存在连续特征, 如何处理:
        会做分桶, 分区操作(区间转换), 之后进行OneHot输入
        数值会不断增加, 对特征影响显著, 如果数值增加, 可以把用户未来信息加入其中, 
        点击率为例: 把历史点击率同意出来作为输入, 统计这个用户对这个商品曝光多少次
                   不能直接使用曝光次数---处理 归一化--->频数转为频率
"""

# 推荐算法 wide_Deep
"""
线性模型和深度模型结合的产物
P(Y = 1|X) = σ(W_wide[X, φ(X)].T + W_Deep.T*α**(l*f) + b)
[X, φ(X)]---->相当于X
会把用户准换成向量---->全连接-需要做embedding
流程:
    LR+DNN(深度神经网络)
    Logistic Loss
        1. ReLU(256) ---> ReLU(512) ---> ReLU(1024) ---> Concatenated Embedding(~1200 dimensions) ---> 离散特征(需要embedding-全连接) & 连续特征
        2. Cross Product Transformation ---> User APP, Impression APP
"""
"wide_Deep model模型实现代码"
# TO EX

# DCN(Deep&Cross)
"""
wide_Deep改进版, 把wide部分LR换成了CrossLayer, 可以显示构造有限阶特征组合, 复杂度低
公式:
    CrossLayer:
        x_(l+1) = x_0 * x_l.T * w_l + b_l + x_l
                = f(x_l, w_l, b_l) + x_l
    Deep:
        ReLU(W_h0x0 + b_h0)
流程:
    1. CrossLayer部分, 不断迭代x_(l+1)进行计算x_Lt
    2. Deep部分, 不断迭代ReLU
    3. concat, x_tack 为两部分--相乘
        Output = Feature Croossing + Bias + Input
    4. p = sigmoid(W_logit*x_stack+b_logit)
"""

# 推荐算法 DeepFM
"""
Deep与FM结合, 是wide_Deep改进版, 将里面LR换成了FM, 增强wide信息提取能力
把每一个特征转为向量, 然后全连接, 最后输出
离散特征: embedding
连续特征: 全连接
流程:
    1. FM:
    sparse Features(iD->向量) --> Embedding / 全连接 ---> FM Layer ---> Output Units
    2. Deep(DNN):
    sparse Features --> Embedding / 全连接 ---> Hidden Layer(激活函数) --> Output Units
提取对应向量(商品序列[1, 3, 4, 5] ---> 转换为128维向量x):
    1. 针对序列中的每个商品id, 通过embedding_lookup方式提取对应的向量(128维), 之后将序列向量求均值
    2. 针对序列中的每个商品id, 通过embedding_lookup方式提取对应的向量(128维), 作为待融合特征向量, 然后提取
       对应预测商品id向量, 通过embedding_lookup方式
       方式:
            将预测商品对应向量作为Q向量, 将序列向通过两个不同全连接后的向量作为k和v向量, 直接基于Q, K, V采用attention
            的思路进行向量融合
    3. 在1基础上, 对embedding_lookup结果基于LSTM做一个特征提取, 将LSTM每个时刻的输出特征做为最终待融合的特征向量
"""

# 向量融合方式
"""
1. 所有向量拼接
    缺点: 要求向量数目必须一样多
2. 直接将所有向量求均值
    缺点: 合并的时候没有偏向
3. 基于Attention的思路进行向量加权合并
    缺点: 计算量多
4. 基于LSTM进行型序列特征提取, 作为最终的向量值
eg:
    样本数据:
        用户点击商品id列表, 当前商品id
    样本1:
        用户点击商品id列表: [1, 3, 5]
            v11: [....] ---> 128维
            v12: [....] ---> 
            v13: [....]
        当前商品id: 2
            p11: [....]
    样本2: 
        用户点击商品id列表: [2, 5, 7, 3]
            v21: [....] ---> 128维
            v22: [....] ---> 
            v23: [....]
            v24: [....]
        当前商品id: 9
            p21: [....]
    样本3: 
        用户点击商品id列表: [1, 3, 5]
            v31: [....] ---> 128维
            v32: [....]
            v33: [....]
        当前商品id: 9
            p31: [....]
        和样本1点击商品id列表完全一样, 只有当前商品id这个特征不一样
    用户点击商品id对应的向量合并:
        1. 样本1: [v11, v12, v13] --> [..., ..., ...] ---> 128维*3 = 384维
           样本2: [v21, v22, v23, v24] --> [..., ..., ..., ...] ---> 512维
        2. 样本1: v1 = (v11 + v12 + v13) / 3 = [...] ---> 128维
           样本2: v2 = (v21 + v22 + v23 + v24) / 4 = [...] ---> 128维向量
           样本3: v3 = (v3的1 + v32 + v33) / 3 = [...] ---> 128维
           此时样本3的点击商品id列表特征对应的向量和样本1的特征向量是同一个
           假设商品映射关系如下:
                1. --> 电脑
                2. --> 鼠标
                3. --> 外套
                4. --> 电源线
                5. --> 裤子
            样本1: [电脑, 外套, 电源线] 鼠标
            样本3: [电脑, 外套, 电源线] 裤子
            从特征混合的希望来讲:
                样本1融合的特征需要更加偏向电脑和电源线
                样本3融合的特征需要更加偏向外套
            公式组合:
                v1 = 0.5*v11 + 0.1*v12 + 0.4*v13
                v3 = 0.04*v31 + 0.9*v32 + 0.06*v33
        3. 用当前商品id对应的特征和列表中每个商品id对应的特征计算相似度, 然后转换成权重系数, 最终加权均值作为最终特征值
           可行的步骤:
            1. 计算当前样本和列表中所有商品id之间的相关性
                s1 = F(p11, v11)
                s2 = F(p11, v12)
                s3 = F(p11, v13)
            2. 将相关性做一个权重转换(softmax)
                a1, a2, a3 = softmax([s1, s2, s3])
            3. 加权均值
                v1 = a1*v11 + a2*v12 + a3*v13
        4. attention
            注意力机制, 将当前关注的特征进行加强, 对不太关注的特征进行减弱;
            通用步骤:
                -1. 计算相关性: 计算当前Q和所有的key之间的相关性
                -2. 相关性转换为权重系数(softmax)
                -3. 加权结果: 将权重系数和所有对应的Value进行加权求和, 结果当做当前输出的V
        5. 以LSTM/RNN/GRU 做核心做序列特征提前
            LSTM里面zhen对每条样本而言, 输入特征的形状/shape必须是[T, E]结构  // T表示时刻长度 E表示每个时刻向量大小
            输入到LSTM的样本转换一下即可:
            样本1: [p11, v11, v12, v13]
            样本2: [p21, v21, v22, v23, v24]
            样本3: [p31, v31, v32, v33] <=> [p31]
"""     

# 推荐算法 xDeepFM
"""
xDeepFM是Wide_Deep的改进版, 在此基础上添加了CIN层(压缩感知层)显式的构造有限阶特征组合. 
xDeepFM与DeepFM两个相关性不大, 与DCN(Deep_Cross)相似
流程:
    对各组向量embedding
        1. 把所有向量拼接输入到DNN里面
        2. 把所有向量拼接输入到linear
        3. 向量与向量之间计算内积--DIN
    输出
公式:
    x^k_(h,*) = Σ_i(1-H_k-1)Σ_j(1-m) (W^(k,h)_ij * (X^(K-1)_(i,*)·X^0_(j,*)))
"""

# 推荐算法 DSSM(召回)
"""
DSSM双塔模型, 主要用来建模每一个query与多个documents的相似度
深度学习-->检索领域
使用物品稀疏特征和用户稀疏特征---最后一层一定是输出向量维度要一样的向量
    即: if user_mpl_units[-1] != spu_mmmllpp_units[-1]
    spu_mlp_units.append(16)
    user_mlp_units.append(16)
def forward(self, spare_x, dense_x):   ---向量内积
    v1 = self.spares_embedding(sparse_x)
    v1 = self.dense'_embedding(dense_x)
    v = torch.cat([v1, v2], dim=1)
    v_size = v.size()
    v = v.view([-1, v_size[1]] * v_size[2]])
    v = self.mlp(v)
    # 由于DSSM模型最终需要计算用户向量和物品向量余弦相似度, 为了线上简化操作, 这里对向量先做一个norm转换
    from torch.nn import functional as F
    v = normalize(v, p=2, dim=1)
    return v
流程:
    文本Q和多个文本D_i -- Trem Vector(n个文本) --> Word Hashing --> similarity(Q·D内积, 余弦相似度) --> probabiliity
定义两个子模型:
    UserModel: 用户侧模型 --- 需要用户稀疏和稠密特征, 内部需要包含特征提取+归一化操作
    ItemModel: 物品侧模型 --- 需要物品稀疏和稠密特征, 内部需要包含特征提取+归一化操作
训练的伪代码:
    user_x = ...  # [N, E1]  # N表示N个样本, E1表示么个样本/用户用E1维向量进行展示
    item_x = ...  # [N, E2]
    target = [0, 1, ...]    # [N] 0表示用户和物品不相关, 1表示相关
    user_feature = UserModel(user_x)    # 提取用户特征, 一般情况下: [N, 128] 
    item_feature = ItemModel(item_x)    # 提取物品特征, 一般情况下: [N, 128] 
    sim = user_feature * item_feature   # [N] 值越大越好对应位置相乘, 数乘, 乘玩还是128
    loss = ...    # 使用sigmoid二分类损失就可
变量命名用户, 物品只是为了区分好理解
用户侧和物品侧
"""
import torch
import torch.nn as nn
l = nn.Linear(100, 2)
d1 = torch.randn(3, 100)
d2 = torch.randn(3, 100)
l(d1)
l(d2)
d = torch.concat([d1[None, ...], d2[None, ...]], dim=0)
d.shape
a = l(d)
a[0]
a[0] - l(d1)  # 参数共享

a = torch.rand(2, 128)
b = torch.rand(2, 128)
c = a*b
c.shape

# 推荐算法 YoutubeNet
"""
主要用于视频推荐, 主要用于召回和精排
特征选择:
    用户历史特征, 上下文特征和交叉特征-->精排
选择内积最大的前N个进行推荐
根据前N个预测下一个视频看哪个类别
流程:
    feature ---> embedding --> average---1. watch vector 2. search vector....--> 全连接ReLU(多次) --> 1. softmax(需要train)--> nearest neighbor index 2. nearest neighbor index
    example age = W_tm(训练时刻固定, 针对所有数据是同一个值)-w_tn(预测标签)---> 物品年龄(虚构特征, 不需要物品实际年龄)
优化策略:
    负采样 --> 海量数据 降计算量
    用之前观测搜索看下一个看哪个视频
"""

# 负采样
"""
目的:
    负采样的目的就是在 negative word 中,找出一部分节点进行权重的更新,而不需要全部都更新
    喜欢的物品我们认为是正样本，用户不喜欢物品我们认为是负样本
    Word2Vec中为了降低复杂度，从全部单词中进行采样（这个过程叫负采样），采出部分单词和目标单词，让模型从中取出预测值最大的单词
功能:
    针对超大类别分类模型的一种简化/优化训练方式, 效果和普通分类模型的训练是一样的
条件:
    1. 类别数目有m个
    2. 批次大小为N
    3. 当前样本属于类别i
网络结构:
    1. 前面都是提取特征属性向量Z的网络结构
    2. 最后一层就是简单的线性转换: S = torch.matmul(Z, W), S的形状是[N, m]表示每个样本属于m个类别的置信度
分类模型的训练方式:
    基于网络最后一层的输出构建交叉损失
        概率的计算:
            p1 = e^S1 / (e^S1 + e^S2 + ... + e^Sm)
            pi = e^Si / (e^S1 + e^S2 + ... + e^Sm)
        损失的计算:
            loss = -ln(pi)
        理解反向传播过程:
            1. 希望loss越小越好 --> pi越大越好
                -1. 分子e^Si越大越好  
                    分母e^S1 + e^S2 + ... + e^Sm 越小越好
                -2. si 越大越好
                    s1, s2, ...., s(i-1), s(i+1)... sm 越小越好
            2. 训练一定是一个不断重复迭代的过程, 不断遍历的过程
                训练方式假设:
                    -1. (本来的分类优化方式)每次遇到当前样本的时候, 都更新所有的S, 满足上述条件
                    -2. 
                        - 在第一次遇到当前样本的时候, 将Si的值增大, 同时将S1, S2, S3, S4减小, 其他Si值不考虑
                        - 在第二次遇到当前样本的时候, 将Si的值增大, 同时将S4, S5, S7, S10减小
                        不断重复上面两个, 只是将减小的S随着批次更改
                        如果训练批次足够多, Si就比所有其他Sj的值都大了  <==> -1.

普通分类模型缺点:
    由于普通分类计算方式是一个线性回归 + softmax概率转换的方式来计算概率, 然后基于概率计算损失你, 最后返回传播
    时间复杂度:
        S = torch.matmul(Z, W) --> Z: [N, 128]   W: [128, m]      
            乘法计算量: N*128*m
        pi = e^Si / (e^S1 + e^S2 + ... + e^Sm)
            计算量: m次
        当类别数目m特别大的时候, 比如: 1亿
    缺点: 当类别数目特别特别大的时候, 正常的噶爱蓝绿经计算的计算量特别大, 模型是无法训练的
负采样:
    思路:
        -1. 没有比你要计算所有负例的置信度值
        -2. 正例的置信度值是每次都要计算的
        -3. 每个批次的训练都是让置信度在局部区域/局部雷本范围内越大越好, 并且通过随机选择负例的类别范围, 通过迭代的思想最终让正例在全局所有类别中也保存最大值
    执行的步骤:
        -1. 从w中抽取正例对应的参数W0
        -2. 从所有负例中随机抽取K个类别, 然后提取对应的参数W1, W2, ...., Wk
        -3. 将所有参数拼接到一起形成一个新的参数U, U的形状就是[128, k+?]   ?可数
        -4. 将原始的标签y映射到新的U参数对应的类别上
        -5. 基于新的参数U做线性转换 + softmmax + 交叉熵损失, 进行参数更新
        计算量只有: N*128*(k+?)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

def t1(h=10, c=10000):
    class Net(nn.module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Embedding(n, 2)
            self.linear = nn.Linear(2, c)

        def forward(self, x):
            x1 = self.embedding(x)
            return self.linear(x1)
    
    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    y0 = torch.tensor([0, 2232, 113, 26, 745, 82])
    net = Net()
    # 得到的是6个样本分别属于c的置信度
    y1 = net(x0)
    # 普通损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y1, y0)
    print(loss)
    # 反向传递
    net.zero_grad()
    loss.backward()
    print(net.linear.weight.grad.size())
    print(net.linear.weight.grad)
    print(net.linear.weight.grad[100: 120])

def t2(n=10, c=10000):
    class NSLayer(nn.Module):
        def __init__(self, in_features, out_features, neg_num):
            super(NSLayer, self).__init__()
            self.out_features = out_features
            self.neg_num = neg_num
            self.weights = Parameter(torch.Tensor(in_features, out_features))
        
        def forward(self, x, y=None):
            if y is None:
                # 和普通线性转换一样, 主要应用在推理Hi好
                return torch.matmul(x, self.weights)
            else:
                # 可以在这里优化
                # 负采样
                # 正例对应权重
                pos_w = self.weights[:, y]
                # 负例对应权重
                neg_idx = np.random.randint(0, self.out_features, self.neg_num*y.size()[0])
                neg_w = self.weights[:, neg_idx]
                # 合并权重
                _w = torch.cat([pos_w, neg_w], dim=1)
                return torch.matmul(x, _w)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Embedding(n, 2)
            self.linear = NSLayer(2, c, 100)

        def forward(self, x, y=None):
            x1 = self.embedding(x)
            return self.linear(x1, y)

    x0 = torch.tensor([1, 2, 3, 6, 7, 8])
    y0 = torch.tensor([0, 2232, 113, 26, 745, 82])   
    net = Net()
    # 普通损失值
    loss_fn = nn.CrossEntropyLoss()   
    # 得到的是6个样本分别属于c的置信度
    y1 = net(x0, y=y0)
    # 更改样本所属类别(和负采样里面属于c个了类别的置信度)
    y0_ = torch.tensor(np.arange(0, y0.size()[0])).long()
    loss = loss_fn(y1, y0_)
    print(loss)
    # 反向传递
    net.zero_grad()
    loss.backward()
    d = net.linear.weight.grad[100: 120].numpy()
    d = np.mean(d, 0)
    d = d[d != 1]
    if len(d) > 1:
        print(net.linear.weight.grad[100: 120])
        if i > 10:
            pass

if __name__ == "__main__":
    t1()

# 推荐算法 BPR
"""
BPR旨在解决物品直接顺序的问题, 提出了一个算法框架可以应用于各个已有推荐算法, 解决物品之间排序问题
思想:
    贝叶斯估计 + SGD随机梯度下降
    偏好物品表示: <u, i, j>  <--->  i > uj
    ex:
        -1. 系统给当前用户曝光的商品id列表为: [p3, p7, p12, p768, p1]
        -2. 用户在曝光列表中点击了商品: p7和p1
        -3. 数据转换
            商品1, 商品2, 偏向选择标签
            p1, p768, 1   偏向于第一个物品
            p1, p3, 1
            p1, p12, 1
            p768, p1, -1    偏向于第二个物品
            p768, p7, -1 
            标签为1表示第一个物品排在第二个物品前, 标签为-1表示第一个物品排在第二个物品后
    x^ = PQ.T
    x^_ui = p_u*q_i.T = Σ(p_uf*q_if)
    x^_uij = x^_ui - x^_uj
    L = Σ -ln(σ(x^_uij)) + λ*1/2*||Θ||**2
    要求:
        希望用户在有偏序商品上的评分差越大越好(sigmoid转换后越接近1越好), 其中Us数据集合就是所有用户u认为物品i比物品j好的一个数据集
        上面来看就是 选择1的数 > -1的数越多越好
    梯度更新Θ:
        Θ = Θ - α*(σL / σΘ)
        σL / σΘ = Σ (σ(x^_uij) - 1) * (σ(x^_uij) / σΘ) + λ*Θ
        σ(x^_uij) / σΘ = σ(p_u*q_i.T - p_u*q_j.T) / σΘ = {1. (q_i - q_j)  if Θ == p_u, 2. p_u  if Θ == q_i,  3. -p_u if Θ == q_j}
"""

# 推荐算法 Redis
"""
Redis管道技术Pipeline是一种最基础优化方法, 不过Pipeline命令太多, 容易造成网络堵塞
Redis采用集群模式, 需要确保所有命令对应的key必须位于同一槽(slot)或者机器上
管道命令模式:
    客户端----> 命令1+命令2 ----> 服务器端 ----> 结果1+结果2 ---> 客户端
危害:
    缓存击穿: 访问非常频繁, 处于集中式高并发访问情况, 大量请求击穿了缓存
    处理: 
        1. 针对不会发生更新的数据可以考虑设置为用不过期
        2. 针对更新不频繁并且缓存刷新耗时少的情况, 可以采用分布式锁或者本地互斥锁保证少量请求能够请求数据库来重新构建缓存, 琦玉线程之际等到锁释放后读取新缓存
        3. 针对更新频繁或者缓存刷新时间长的情况, 采用定时线程在过期前主动构建缓存或者延后缓存过期时间
    缓存穿透:
        异常请求直接无法从缓存中提取数据, 每次都需要直接查询数据库情况
    处理: 在数据库中没有查到的数据添加一个特殊值
    缓存雪崩:
        缓存服务器突然宕机4或者大量key同时过期, 导致所有的操作均落在数据库上
    处理:
        1. 事前: redis高可用, 主从+哨兵, redis cluster, 避免全盘崩溃, 缓存过期时间添加随机值
        2. 事中: 本地缓存 + 服务限流&降级, 避免 MySQL 被打死
        3. 事后: redis持久化, 一旦重启, 自动从磁盘上加载数据, 快速恢复缓存数据
"""

# faiss_demo 基础构建
import time
import faiss
import numpy as np

if __name__ == '__main__':
    np.random.seed(10)
    # 模拟一个10w个128维向量
    xb = np.random.randn(50000, 128).astype('float32')    # 向量库 必须float32
    x1 = np.random.randn(100, 128).astype('float32')     # 新增商品向量
    x0 = np.random.randn(1, 128).astype('float32')    # 待检索的向量

    # faiss的使用
    dim, measure = 128, faiss.METRIC_L2     # .MMETRIC_INNER_PRODUCT余弦   .METRIC_L1
    """
    这里为暴力检索 还有1. IVFx Flat 倒排索引+暴力检索, 先聚类, 得到x个簇, 然后每个簇暴力检索; 2. PQx 乘积量化, 将向量维度切成x段, 每段检索, 最后取交集得到TopK, 召回率相对较高;
    3. IVFx PQy 工业界大量使用, 乘积量化, 改进IVF的k-means, 将一个向量维度切成x段, 每段分别进行k-mmeans再检索; 4. HNSWx 图检索方法, 时间复杂度低, 支持分批导入, 极其适合线上任务, 但占用内存大
    注: 这里的x 可以换为 32 64 128位数
    """
    param = 'Flat'      
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)     # 索引模型是否已经训练好, 如果为False, 需要单独调用train方法
    print(type(index))
    index.add(xb)   # 添加
    print(index.is_trained)
    t1 = time.time()
    print(index.search(x0, 5))      # 第一部分是距离, 第二部分是最相似度的index下标
    print(time.time() - t1)

    '参数改变'
    dim, measure = 128, faiss.METRIC_L2 
    param = 'HNSW64'       # 新加入一个点和最近64个点相邻
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)     # 索引模型是否已经训练好, 如果为False, 需要单独调用train方法
    if not index.is_trained:
        index.train(xb)
    print(type(index))
    index.add(xb)   # 添加
    print(index.is_trained)
    t1 = time.time()
    print(index.search(x0, 5))      # 第一部分是距离, 第二部分是最相似度的index下标
    print(time.time() - t1)
    index.add(x0)
    index.add(x1)
    print(index.search(x0, 5))

# 推荐算法 Faiss_HNSWx
"""
HNSW: 可导航小世界网络算法,
特点:
    1. 任意节点之间均可通过直接或者间接方式连接起来, 不存在孤立点
    2. Short-Range Edge用于连接两个相近的点
    3. Long-Range Edge---> 高速公路, 用于连接两个距离较远的点, 可以直接相连, 不需要转换
    4. 跳表结构: 一种快速查询数据的一种方式
    5. HNSW是在NSW的基础上发展过来的(加跳表结构)
数据查找:
    节点插入后, 每层节点随机抽样, 从最高层逐步向下寻找正确顺序, 找邻居, 如果没有, 往下一层, 一层一层找邻居, 直到找到最近的邻居
"""

# 推荐系统项目需求结构
"""
大方向:  1. 召回   2. 过滤    3. 精排
流程:
    1. 提取当前接口对应规则 也就是具体召回策略, 过滤策略, 精排策略
    2. 基于召回策略从各种数据源(redis, MySQL)提取商品初始列表
        如果存在多个召回策略, 实际上这些召回是并行的, 最终召回商品序列实际上就是所有召回列表合并
    3. 基于过滤策略定义的规则, 对初始商品列表进行过滤拦截
        存在多个过滤测试, 那边相当于这些过滤必须都满足
    4. 调用精排策略里面定义的策略模型对商品序列进行排序
        如果存在多个精排策略, 那么多个策略之间乘法合并关系
    5. 获取TopN商品列表返回给调用方完成推荐
"""

# 召回策略
"""
多路召回策略--商品: 
u2i: 基于用户推荐对应的商品列表---> 协同过滤, 矩阵分解
i2i: 基于物品推荐物品的商品列表, 基于物品推荐相似物品---> 相似度(相似, 协同过滤, 矩阵分解)
u2ui: 首先基于用户获取当前用户的相似用户, 然后将相似用户偏好/推荐的结果作为当前用户的推荐结果
u2i2i: 首先基于用户获取到商品列表, 然后基于物品推荐相似物品
    - 模型相关类:
    1. UserCF: 基于用户的协同过滤算法为每个用户产生一个推荐商品列表
    2. ItemCF: 基于物品的系统过滤算法为每个用户产生一个推荐商品列表
    3. MF: 基于矩阵分解算法为每个用户产生一个推荐商品列表
        离线部分:
        -1. 基于用户协同过滤算法为每个用户产生一个推荐商品列表
        -2. 构建最优的UserCF, ItemCF, MMF模型, 过程中可能存在GridSearch网格餐胡选择, 模型效果评估
        -3. 基于模型对每个用户产生一个推荐列表/召回商品列表, 根据需要决定时是否需要将评分过的商品进行删除
        -4. 将产生的商品列表保存到redis数据库中, 采用hash结果存储数据
            key: 'rec': recall:u2i:user_id'其中user_id就是具体用户id
            field name: 具体user2time推荐算法类型, eg: usercf, itemcf, mf等
            field value: 推荐商品列表字符串, 格式为: 'pid: scoorel, pid2: score2
        -5. 模型更新:
            一般情况基于用户物品矩阵模型一天一更, 小流程跑一遍
            如何一天一更: 采用定时回归任务--> 基于各种框架, 基于Linux服务器默认线上部分 
        线上部分:
        -1. 直接从redis直接提取数据就行
        -2. 召回策略为: 'i2i: views: xxxx': xxxx就是具体是算法, 也就是redis中具体field name可以有多个
    4. UserViewItemSim: 基于用户最近浏览商品相似商品召回
        - 上面基于用户评分矩阵构建模型
    5. FM: 基于FM是算法为每个用户产生一个推荐商品列表
        第一层: 通过embedding转换成向量
                稀疏提取
                self.embedding = nn.Embedding(num_embeddings=sum(field_dims), embedding_dim=embed_dim)
                稠密提取---直接全连接
                self.linear = nn.Linear(1, out_features, bias=False)
    6. Youtube Net/DSSM向量召回: 基于商品向量相似来召回视频列表
    7. DeepFM精排模型: 基于DeepFM深度学习模型进行推荐商品精排
    8. 模型策略:    LR, GBDT+LR, FM, BPR四种基础机器学习模型
        GBDT+LR和LR相似, 只是在其中加入GBDT处理特征--流程:
            先做叶子分类:
                特征转换:
                x_df, label_encoders = self.prepare_data(x_df)
                模型训练:
                gbdt = GrandiientBoostingClassifier(n_estimators=100, max_depth=3)
                gbdt.fit(x, y)
                x = gbdt.apply(x)  获取叶子节点位置信息
                x = np.reshape(x, (x.shape[0], -1))
                onehot = prepreocessing.OneHotEncoder(handle_unknown='ignore')
                x = onehot.fit_transform(x)
                lr = Ridge()
                lr.fit(x, y)
                y_ lr.predict(x)
        BPR:
            def __init__(self, uesr_size, spu_size, embed_dim=4):
                self.w = nn.Parameter(torch.empty(user_size, embed_dim))    # 用户向量矩阵
                self.h = nn.Parameter(torch.empty(user_size, embed_dim))    # 商品向量矩阵
                nn.iniit.xavier_normal_(self.w.data)
                nn.iniit.xavier_normal_(self.h.data)

            def forward(selfm u, i, j):
                u = self.w[u, :]    # [batch_size, embed_dim]
                i = self.h[i, :]
                j = self.h[j, :]
                r_ui = torch.mul(u, i).sum(dim=1)    # 数乘+求和 -> 用户u对应物品i的评分
                r_uj = torch.mul(u, j).sum(dim=1)    # 数乘+求和 -> 用户u对应物品j的评分
                r_uij = r_ui - r_uj     
                # 损失越小 r_uij越大越好
                log_prob = -F.logisgmoid(r_uij).sum()
                # l2损失
                _alpha_w = 0.1
                _alpha_h = 0.1
                # l2_Loss
                l2_loss = _alpha_w * torch.norm(u) + _alpha_h * torch.norm(i) + _alpha_h * torch.norm(j)
            def predict(self, u, i):
                u = self.w[u, :]
                i = self.h[i, :]
                r_ui = torch.mul(u, i).sum(dim=1)
                return r_u
            模型加载
            self.user_id_mapping = SimpleMapping(os.path.join(model_diir, 'bpr_dict', "user_id.dict"))
            self.spu_id_mapping = SimpleMapping(os.path.join(model_diir, 'bpr_dict', "user_id.dict"))
        - 基于用户物品特征属性矩阵, 构建对应机器学习, 深度学习模型即可
            离线部分:
                -1. 从数据库提取数据
                -2. 数据处理/特征工程/离线模型训练/离线模型评估/模型持久化
            模型上线部署:
                -1. 
                    方式1: 云平台部署: PAI/Sagemaker
                    方式2: 直接在服务器内部进行模型恢复加载
                -2.
                    在推荐服务器内部实现调用模型的相关code
    - 规则类型类:
    1. 新品召回: 针对最近添加的100个最新商品作为推荐商品列表
    2. 热销品召回: 针对最近浏览次数最多的100个商品作为推荐商品列表
    3. 分地域新品召回: 针对用户对应地域最近添加的100个最新商品作为推荐商品列表
    4. 分地域热销品召回: 针对用户对应地域最近浏览次数最多的100个商品作为推荐商品列表
        - 基于物品, 用户的本身特征信息进行对应的规则逻辑代码处理
        - 基于规则进行对应的逻辑代码处理即可
        - 直接基于规则从数据库提取数据
多路召回策略--视频:
    1. Youtube Net/DSSM向量召回: 基于商品向量相似来召回视频列表
    2. 同类新品召回: 针对当前视频同一品类的其他视频作为召回商品列表
    3. 同品类热映召回: 针对当前视频同一品类的其他热映视频作为召回商品列表
    4. 类似品类新品召回: 针对当前视频类似品类的其他视频作为召回商品列表
    5. 类似品类热映品召回: 针对当前视频类似品类的其他热映视频作为召回商品列表
    6. 参演标签召回: 基于电影参演者标签提取所有对应视频-->提取某明星参演
"""

# 过滤策略
"""
过滤方式(商品+视频):
    1. UserViewFilter: 用户最近观看过的10个视频需要过滤
    2. BlacklsFillter: 黑名单过滤, 针对用户标注不喜欢的视频全部过滤
    3. UserNoLikeFilter: 用户不喜欢类型时佩璞过滤
实现流程:
    基于同召回策略, 将相关过滤信息持久化到redis中 
    然后过滤策略中先从redis中提取相关规则信息
    然后再按照过滤规则进行数据过滤即可
"""

# 精排策略
"""
精排策略:
    1. 模型策略: LR, GBDT+LR, FM, BPR四种基础机器学习模型
    2. 规则精排: 用户偏好类别视频加权, 新视频加权
实现流程:
    相关规则信息持久化到redis中
    然后精排策略先从redis中提取相关规则信息
    之后再按照规则进行数据排序即可

精排策略(视频):
    1. DeepFM精排模型: 基于DeepFM深度学习模型进行推荐序列商品精排
    2. 规则精排模型: 品类加权, 时间加权
"""

# 策略模型改变评估---AB测试, 部署, 更新
"""
模型更新直接上线
模型策略更改:
    1. 线下(AUC)  2.线上(AB_test) 
AB_test:
    数据分发:
        目前使用策略1, 2, 3----1, 2 为老模型, 流量数据分发, 80%策略2上, 20%策略3上
        CTR(2点击数量应该是3的四倍):
            策略2: <0.03 模型效果不好  < 4w
            策略3: 为0.03  1w
            则新模型效果好
        目前使用策略1, 流量数据分发, 50%策略2上, 50%策略3上
            策略2: 2w
            策略3: 3w
            则新模型效果好
        模型调换:
        目前使用策略1, 流量数据分发, 50%策略2上, 50%策略3上
            策略3: 3w
            策略2: 2w
            则新模型效果好
        判断是否为用户偏差, 排除后----> 新模型有优势----> 切换模型 
"""

# DIN_Deep
"""
利用物品相关性权重来推荐
CTR深度学习模型固定模式:
    高维稀疏特征映射低维embedding向量, embedding向量通过组合转化为固定长度向量, 
    再将所有特征属性向量拼接传入全连接层
流程:
    用户基础特征 --> concat -----------> concat ---> PReLU(200) ---> PReLU(80) ---> softmax(2) ---> output
    物品 ---> sum pooling ---> concat -↑
"""
class Dice(nn.Module):

    def __init__(self) -> None:
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, )))
    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x)
        return x.mul(p) + self.alpha * x.mul(1 - p)

'''偏理论代码'''
import torch
import torch.nn as nn

class Basicdin(nn.Module):
    def __init__(self) -> None:
        super(Basicdin, self).__init__()
        embeddiing_size = 88
        self.user_profile_embedding_layer = SparseFeaturesEmbedding(
            field_dims = [2, 10],   # 给定用户基础特征每个特征属性的类别数目列表, 总特征数目是C1=len(field_dimsd5)
            embed_dim = embeddiing_size
        )   # 这个函数是自己写
        self.ad_embedding_layer = SparseFeaturesEmbedding(
            field_dims = [10^9, 10^7, 10^5],   # 给定物品基础特征每个特征属性的类别数目列表, 长一d为3: 商品id, d铺id, 品类id
            embed_dim = embeddiing_size
        )   # 这个函数是自己写
        self.context_embedding_layer = SparseFeaturesEmbedding(
            field_dims = [10, 10],   
            embed_dim = embeddiing_size
        )   # 这个函数是自己写
        self.mlp = MultilayerPerceptron(input_features=80, units=[200, 80, 2])

    def forward(self, user_profile_features, user_behaviors, candidate_ad_feature, context_features):
        """
        基于输入d四个方面特征信息, 进行前向过程, 最终输出d于候选商品d击, 不d击置信d
        N 表示每个批次样本d小
        :param user_profile_features: [N, C1]用户基础特征, 假d全是离散特征, 每个用户用C1个int类型d id值进行特征描
        :param user_behaviors: [N, T, 3]用户d行为特征, 每个用户d 存在一个行为序列(长为T), 序列内每个时刻d应操作包括三个特征(d铺id, 种类id, 商品id)
        :param candidate_ad_feature: [N, 1, 3]候选商品基础特征, 包括: 商品id, d铺id, 品类id
        :param context_features:
        """
        # 1. embedding
        batch_size = user_profile_features.size()[0]
        l2_loss = []

        user_features = self.user_profile_embedding_layer(user_profile_features)
        user_features = user_features.view(user_features.size()[0], -1) # [N, C1, E] -> [N, C1*E]
        l2_loss.append(user_features)
        user_behavior_features = self.ad_embedding_layer(user_behaviors) # [N, T, 3] ---> [N, T, 3, E]
        behavior_length = user_behavior_features.size()[1]
        user_behavior_features = user_behavior_features.view(batch_size, behavior_length, -1) # [N, C1, E] -> [N, T*3*E]
        l2_loss.append(user_behavior_features)
        ad_feature = self.ad_embedding_layer(candidate_ad_feature) # [N, 1, 3] -> [N, 1, 3, E]
        ad_feature = ad_feature.view(user_features.size()[0], -1) # [N, C1, E] -> [N, 1*3*E]
        l2_loss.append(ad_feature)
        context_features = self.context_embedding_layer(context_features) # [N, C2] -> [N, C2, E]
        context_features = context_features.view(batch_size, -1)    # [N, C2, E] ---> [N, C2*E]
        l2_loss.append(context_features)

        # 2. concat 合并embedding结果 stack/cat
        x = torch.cat([user_features, user_behavior_features, ad_feature, context_features], dim=1)

        # 3.全连接
        x = self.mlp(x) 
    
        # 4.计算损失
        if return_l2_loss:
            l2_loss = sum(torch.pow(v, 2).sum() for v in l2_loss)
            l2_loss += sum([torch.pow(v, 2).sum() for v in self.mlp.parameters()])
            return x, l2_loss
        else:
            return x

if __name__ == "__main__":
    m = Basicdin()
    r = m(
        user_profile_embedding_layer = torch.tensor([
            [0, 7],
            [1, 8]
        ]).int(),
        user_behaviors = None, 
        candidate_ad_feature = torch.tensor([
            [
                [21, 15, 31],
                [15, 22, 22],
                [52, 77, 5],
                [23, 34, 612]
            ],
            [
                [897, 23, 12],
                [534, 2123, 223],
                [522, 737, 555],
                [2123, 34, 612]
            ]
        ]),
        context_features = torch.tensor([
            [2, 4],
            [5, 3],
            [5, 3]
        ]).int()
    )
    print(r.shape)

# DIEN_Deep
"""
用户行为序列建模, 融合方式, 均值加权求和合并
没有考虑用户兴趣转移变化
要求提取出来向量和下一个向量接近
不同: 引入GRU结构
流程:
    用户行为特征 --> embedding -----------> concat ---> PReLU(200) ---> PReLU(80) ---> softmax(2) ---> output
    物品 ---> sum pooling ---> concat -↑
"""
class Dien(nn.Module):
    def __init__(self) -> None:
        super(Dien, self).__init__()
        embedding_size = 88
        self.user_profile_embedding_layer = SparseFeaturesEmbedding(
            field_dims = [2, 10],   # 给定用户基础特征每个特征属性的类别数目列表, 总特征数目是C1=len(field_dimsd5)
            embed_dim = embedding_size
        )   # 这个函数是自己写
        self.ad_embedding_layer = SparseFeaturesEmbedding(
            field_dims = [10^9, 10^7, 10^5],   # 给定物品基础特征每个特征属性的类别数目列表, 长一d为3: 商品id, d铺id, 品类id
            embed_dim = embedding_size
        )   # 这个函数是自己写
        self.context_embedding_layer = SparseFeaturesEmbedding(
            field_dims = [10, 10],   
            embed_dim = embedding_size
        )   # 这个函数是自己写
        self.mlp = MultilayerPerceptron(input_features=80, units=[200, 80, 2])
        self.gru = nn.GRU(input_size=3 * embedding_size, hidden_size=3 * embedding_size, num_ayers=1, batch_first=True)
    
    def forward(self, user_profile_features, user_behaviors, candidate_ad_feature, context_features):
        """
        基于输入d四个方面特征信息, 进行前向过程, 最终输出d于候选商品d击, 不d击置信d
        N 表示每个批次样本d小
        :param user_profile_features: [N, C1]用户基础特征, 假d全是离散特征, 每个用户用C1个int类型d id值进行特征描
        :param user_behaviors: [N, T, 3]用户d行为特征, 每个用户d 存在一个行为序列(长为T), 序列内每个时刻d应操作包括三个特征(d铺id, 种类id, 商品id)
        :param candidate_ad_feature: [N, 1, 3]候选商品基础特征, 包括: 商品id, d铺id, 品类id
        :param context_features:
        """
        # 1. embedding
        batch_size = user_profile_features.size()[0]
        l2_loss = []

        user_features = self.user_profile_embedding_layer(user_profile_features)
        user_features = user_features.view(user_features.size()[0], -1) # [N, C1, E] -> [N, C1*E]
        l2_loss.append(user_features)
        user_behavior_features = self.ad_embedding_layer(user_behaviors) # [N, T, 3] ---> [N, T, 3, E]
        behavior_length = user_behavior_features.size()[1]
        user_behavior_features = user_behavior_features.view(batch_size, behavior_length, -1) # [N, C1, E] -> [N, T*3*E]
        l2_loss.append(user_behavior_features)
        ad_feature = self.ad_embedding_layer(candidate_ad_feature) # [N, 1, 3] -> [N, 1, 3, E]
        ad_feature = ad_feature.view(user_features.size()[0], -1) # [N, C1, E] -> [N, 1*3*E]
        l2_loss.append(ad_feature)
        context_features = self.context_embedding_layer(context_features) # [N, C2] -> [N, C2, E]
        context_features = context_features.view(batch_size, -1)    # [N, C2, E] ---> [N, C2*E]
        l2_loss.append(context_features)

        # 特征融合&加权
        user_behavior_features, _ = self.gru(user_behavior_features)
        user_behavior_features = self.act_unit_layer(user_behavior_features, ad_feature)
        user_behavior_features = user_behavior_features.sum(dim=1)      # [N, T, 3*E] ---> [N, 3*E]
        _, user_behavior_features = self.gru(user_behavior_features)

        # 2. concat 合并embedding结果 stack/cat
        x = torch.cat([user_features, user_behavior_features, ad_feature, context_features], dim=1)

        # 3.全连接
        x = self.mlp(x) 
    
        # 4.计算损失
        if return_l2_loss:
            l2_loss = sum(torch.pow(v, 2).sum() for v in l2_loss)
            l2_loss += sum([torch.pow(v, 2).sum() for v in self.mlp.parameters()])
            return x, l2_loss
        else:
            return x

if __name__ == "__main__":
    m = Basicdin()
    r = m(
        user_profile_embedding_layer = torch.tensor([
            [0, 7],
            [1, 8]
        ]).int(),
        user_behaviors = None, 
        candidate_ad_feature = torch.tensor([
            [
                [21, 15, 31],
                [15, 22, 22],
                [52, 77, 5],
                [23, 34, 612]
            ],
            [
                [897, 23, 12],
                [534, 2123, 223],
                [522, 737, 555],
                [2123, 34, 612]
            ]
        ]),
        context_features = torch.tensor([
            [2, 4],
            [5, 3],
            [5, 3]
        ]).int()
    )
    print(r.shape)

# DSIN_Deep
"""
拟合CTR预测任务中用户行为, 将用户行为分为成会话 
然后使用bias偏度的self-attention对每个会话建模提取用户兴趣
然后用Bi-LSTM捕捉用户不同历史会话兴趣之间交互
最后用局部激活单元对用户兴趣特征进行聚焦
"""

# FiBiNet
"""
特征重要性选择和特征交互能力的网络结构
特征交互:
    一部分对embedding之后内容做交互, 一部分SENet加权之后特征做交互
    第一个特征向量和第二个特征向量做交互
    交互向量共享W
"""

# 多目标算法
"""
除了有CTR点击率预估外, 还有衍生YSL估计----预估第二个页面点击率, 预估查看购物车, 预估查看商品下单率, 预估视频点击后观看比率
实现方法:
    分为多目标构建不同模型, 然后在精排阶段将多目标预测结果融合
    样本loss加权---将其他转化为样本权重---改变数据分布达到其他优化效果---权重系数需要通过线上ab_test计算确定
    多任务学习: Shared-Bottom Multi-task Model  Moe  MMOE(企业最推荐)  ESMM  PLE---
    在置信度位置进行操作, 例如乘价格等等, 负采样后--需要做校准
Shared-Bottom Multi-task Model:
    通过浅层神经网络参数实现互补学习  输入--->Shared-Bottom--两个子网络(可以用随意选lstm, bert, cnn)--> {1. Tower A: 输出A   2. Tower B: 输出B}---> 概率值相乘
    两个任务不能出现互斥行为---相关性高学习好
    cv里面也能用
Moe:
    通过门结构--->特征选择
    输入--->{gate: 1. expert0 2. expert0 3. expert0} --融合(加权求和)-> 两个子网络--> {1. Tower A: 输出A   2. Tower B: 输出B}
MMOE:
    为每个任务创建一个门结构
    输入--->{1.gateA:  1. expert0 2. expert0 3. expert0,  2.gataB: 1. expert0 2. expert0 3. expert0} --融合(加权均值求和)-> 两个子网络--> {1. Tower A: 输出A   2. Tower B: 输出B}
ESMM:
    任务依赖提出---电商推荐多目标预估经常是ctr和cvr
PLE:
    将共享部分和特定部分显式分开
""" 
# Shared-Bottom Multi-task Model:代码结构
import torch.nn as nn
import os
import torch
class Export(nn.module):
    def __init__(self) -> None:
        super(Export, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class SBModule(nn.module):
    def __init__(self) -> None:
        super(SBModule, self).__init__()
        self.export = Export()
        self.task1_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
        self.task2_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # 1. 提取共享特征
        z = self.export(x)
        # 2. 分任务输出
        o1 = self.task1_layer(z)
        o2 = self.task2_layer(z)
        return o1, o2

def test():
    m = SBModule()
    x = torch.randn(8, 17)  # 8个样本, 每个样本17个原始输入特征
    y = m(x)
    # 可视化输出
    mode = torch.jit.trace(
        m.cpu().eval(),
        example_inputs=(x.cpu,)
    )
    torch.jit.save(mode, os.path.join('.', 'Shared-Bottom.pt'))
    return y

if __name__ == '__main__':
    test()


# MOE:代码结构
import torch.nn as nn
import os
import torch
class Export(nn.module):
    def __init__(self) -> None:
        super(Export, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class Gate(nn.Module):
    def __init__(self, n) -> None:
        super(Gate, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(16, n),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layer(x)

class MOE(nn.module):
    def __init__(self) -> None:
        super(MOE, self).__init__()
        self.export = Export()
        self.exports = nn.ModuleList([
            Export(), Export(), Export(), Export()
        ])
        self.gate = Gate(n = 4)
        self.task1_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
        self.task2_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # 1. 提取共享特征
        w = self.gate(x)   # [N, 4] 权重
        '进行值分辨'
        z = [w[:, i:i+1] * self.exports[i](x) for i in range(4)]
        z = sum(z)
        # 2. 分任务输出
        o1 = self.task1_layer(z)
        o2 = self.task2_layer(z)
        return o1, o2

def test():
    m = MOE()
    x = torch.randn(8, 17)  # 8个样本, 每个样本17个原始输入特征
    y = m(x)
    # 可视化输出
    mode = torch.jit.trace(
        m.cpu().eval(),
        example_inputs=(x.cpu,)
    )
    torch.jit.save(mode, os.path.join('.', 'moe.pt'))
    return y

if __name__ == '__main__':
    test()


# MMOE:代码结构
import torch.nn as nn
import os
import torch
class Export(nn.module):
    def __init__(self) -> None:
        super(Export, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class Gate(nn.Module):
    def __init__(self, n) -> None:
        super(Gate, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(16, n),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        z = self.layer(x)
        z = z.view(-1, 1, n)
        return z

class MMOE(nn.module):
    def __init__(self) -> None:
        super(MMOE, self).__init__()
        self.export = Export()
        self.exports = nn.ModuleList([
            Export(), Export(), Export(), Export()
        ])
        self.gate1 = Gate(n = 4)
        self.gate2 = Gate(n = 4)
        self.task1_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
        self.task2_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # 1. 提取共享特征
        w1 = self.gate1(x)   # [N, 1, 4] 权重
        w2 = self.gate2(x)   # [N, 1, 4] 权重
        '进行值分辨'
        z = [self.exports[i](x) for i in range(4)]   # list, 里面有4个tensor, 每个tensor形状为[N, 64]
        z = torch.cat(z, dim=-1) # [N, 64, 4]
        z1 = (w1 * z).sum(dim=-1)
        z2 = (w2 * z).sum(dim=-1)

        # 2. 分任务输出
        o1 = self.task1_layer(z1)
        o2 = self.task2_layer(z2)
        return o1, o2

def test():
    m = MMOE()
    x = torch.randn(8, 17)  # 8个样本, 每个样本17个原始输入特征
    y = m(x)
    # 可视化输出
    mode = torch.jit.trace(
        m.cpu().eval(),
        example_inputs=(x.cpu,)
    )
    torch.jit.save(mode, os.path.join('.', 'mmoe.pt'))
    return y

if __name__ == '__main__':
    test()


# PLE:代码结构
import torch.nn as nn
import os
import torch
class Export(nn.module):
    def __init__(self) -> None:
        super(Export, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class Gate(nn.Module):
    def __init__(self, n) -> None:
        super(Gate, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(16, n),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        z = self.layer(x)
        z = z.view(-1, 1, n)
        return z

class PLE(nn.module):
    def __init__(self) -> None:
        super(PLE, self).__init__()
        '''分成3个export组合'''
        self.task1_export = nn.ModuleList(
            Export(), Export(), Export(), Export()
        )
        self.task2_export = nn.ModuleList(
            Export(), Export(), Export(), Export()
        )
        self.shared_export = nn.ModuleList(
            Export(), Export(), Export(), Export()
        )
        self.gate1 = Gate(n = 8)
        self.gate2 = Gate(n = 8)
        self.task1_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
        self.task2_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # 1. 提取共享特征 --- 12个分支
        w1 = self.gate1(x)   # [N, 1, 8] 权重
        w2 = self.gate2(x)   # [N, 1, 8] 权重
        '进行值分辨'
        task1_z = [self.task1_export[i](x) for i in range(4)]   # list, 里面有4个tensor, 每个tensor形状为[N, 64]
        task2_z = [self.task2_export[i](x) for i in range(4)]
        shared_z = [self.shared_export[i](x) for i in range(4)]
        task1_z.extend(shared_z)
        task1_z = torch.cat(task1_z, dim=-1)
        task1_z = (w1 * task1_z).sum(dim=-1)
        task2_z.extend(shared_z)
        task2_z = torch.cat(task2_z, dim=-1)
        task2_z = (w2 * task1_z).sum(dim=-1)
        # 2. 分任务输出
        o1 = self.task1_layer(task1_z)
        o2 = self.task2_layer(task2_z)
        return o1, o2

def test():
    m = PLE()
    x = torch.randn(8, 17)  # 8个样本, 每个样本17个原始输入特征
    y = m(x)
    # 可视化输出
    mode = torch.jit.trace(
        m.cpu().eval(),
        example_inputs=(x.cpu,)
    )
    torch.jit.save(mode, os.path.join('.', 'ple.pt'))
    return y

if __name__ == '__main__':
    test()


# 搜索体系结构
"""
与推荐相似
必须基于keyword 做召回
DSSM
包括:
    增加特征--搜索词 keyword
    召回, 过滤, 精排, 重排
"""