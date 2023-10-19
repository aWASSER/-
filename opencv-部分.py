# OpenCV基础
# 基本处理
"""
默认[R, G, B] 顺序变为特征为[G, B, R] 顺序
可视化:
    cv.imshow()
"""
import cv2 as cv
import numpy as np
import matplotlib as plt
# 图像读取
"""
imread: 默认返回结果是: [H, W, C]加载返回通道是: BGR
"""
img = cv.imread('xxx.png', 0)    # 第一个参数 图片路径, 第二个参数 图像读取方式  0 表示灰度图像加载, 1表示BGR图像, 默认1,  -1表示加载alpha透明通道图像
print(type(img), img.shape)
print(np.shape(img))

img[: : 0]  # B通道
img[: : 1]  # G 通道

# 图像展示
cv.imshow('image')
# 让图像暂停dealy秒, 当delay秒设置为0的时候, 表示永远, 当键盘任意输入时候, 结束暂停
cv.waitKey(0)
# 释放所有资源
cv.destroyAllWindows()

# 图像保存
cv.imwrite('t1.png', img)

# 根据不同输入进行图像操作
cv.imshow('image', img)
# 等待键盘输入
k = cv.waitKey(0)    # & OxFF

if k == 27:
    print(k)

# 读取图像将图像转化为matplotlib可视化----改图像颜色
img = cv.imread('t1.png', 1)
img2 = np.zeros_like(img, dtype=img.dtype)
img2[:, :, 0] = img[:, :, 2]
img2[:, :, 1] = img[:, :, 2]
img2[:, :, 1] = img[:, :, 2]
print(img2.shape)
plt.imshow(img)
plt.show()

# 改变形状
img = cv.imread('t1.png', 1)
imgl = np.transpose(img, (2, 1, 0))
# 这个transpose操作实际上更改维度顺序, 也就是将img的[600, 510, 3]这个形状转换为[3, 510, 600]

# numpy处理
# 可视化
cv.imshow('image', np.transpose(img, (1, 0, 2)))
# 让图像暂停delay毫秒
cv.waitKey(0)
# 释放所有资源
cv.destroyAllWindows()

# 视频基本处理
# 从摄像头获取视频
# 创建一个基于摄像头的视频读取, 给定基于第一个视频设备
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
capture = cv.VideoCapture(0)
# 设置摄像头相关参数(实际会有偏移)
success = capture.set(cv.CAP_PROP_POS_FRAMES, 800)
if success:
    print('设置宽度成功')
success = capture.set(cv.CAP_PROP_POS_FRAMES, 480)
if success:
    print('设置高度成功')
# 打印属性
size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)
# 遍历获取视频中图像
# 读取摄像头获取图像
success, fram = capture.read()
# 遍历以及等待任意键盘输入(-1表哦是等待delay后, 没有任何键盘输入)
while success and cv.waitKey(1) == -1:
    cv.imshow('fram',fram)
    # 读取下一帧图片
    success, frame = capture.read()
# 释放资源
capture.release()
cv.destroyAllWindows()

# 基于opencv基本绘画
# 创建一个黑色的图像
img = np.zeros((512, 512, 3), np.uint8)
# 画一条线
cv.line(img, pt1=(0, 0), pt2=(511, 511), color=(255, 0, 0))
# 可视化
cv.imshow('image', img)
cv.waitKey(0)
# 释放窗口资源
cv.destroyAllWindows()

# 创建一个黑色的图像
img = np.zeros((512, 512, 3), np.uint8)
# 画一个矩形
cv.rectangle(img, pt1=(10, 10), pt2=(50, 320), color=(255, 0, 0))
# 可视化
cv.imshow('image', img)
cv.waitKey(0)
# 释放窗口资源
cv.destroyAllWindows()

# 创建一个黑色的图像
img = np.zeros((512, 512, 3), np.uint8)
# 画一个椭圆: 圆心, 轴长, 偏移角度, 椭圆的角度信息 -- 有问题
# thickness=-1为填充, 其他为线
cv.ellipse(img, pt1=(210, 310), axes=(100, 50), angle=0, startAngle=360, color=(0, 0, 255), thickness=-1)
cv.ellipse(img, pt1=(410, 410), axes=(50, 50), angle=0, startAngle=180, color=(0, 0, 255), thickness=2)
cv.rectangle()
# 可视化
cv.imshow('image', img)
cv.waitKey(0)
# 释放窗口资源
cv.destroyAllWindows()

# 画多边形
# 创建一个黑色的图像
img = np.zeros((512, 512, 3), np.uint8)
# 画一个多边形 -- 有问题
cv.polylines(img, pt1=(10, 10), pt2=(50, 320), color=(255, 0, 0))
# 可视化
cv.imshow('image', img)
cv.waitKey(0)
# 释放窗口资源
cv.destroyAllWindows()

# 绘制文字
# 创建一个黑色图像
img = np.zeros((512, 512, 3), np.uint8)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, text='Awasser', org=(10, 450), fontFace=font, 
            fontScale=4, color=(255, 255, 255), thickness=2, lineType=cv.LINE_AA)
# 可视化
cv.imshow('image', img)
cv.waitKey(0)
# 释放指定窗口资源
cv.destroyAllWindows()

# OpenCV算术运算
"""
img.item   获取对应列, 对应行像素值
"""
import cv2 as cv
import numpy as np
import matplotlib as plt

# 截图
'访问图像像素--(行, 列)'
px = img[255, 300]
print(f'位置(250, 300)对应像素为: {px}')
blue = img[250, 300, 0]
print(f'位置(250, 300)对应的蓝色像素为: {blue}')
# 甚至所有红色像素为127
img[:, :, 2] [127]
print(f'位置(250, 300)对应像素为: {img[250, 300]}')

# 基于image对象获取对应像素值
print(f'位置(250, 300)对应的蓝色像素为: {img.item(250, 300, 0)}')
# 设置像素值--取色
img.itemset((250, 300, 0), 100)
print(f'位置(250, 300)对应像素的蓝色取值为: {img.item(250, 300, 0)}')

img = cv.imread('t.png')
# 图像内部粘贴
box = img[0:95, 20:240]
box2 = img[0:95, 280:500]
# img[0:95, 280:500] = box
box2 = box2 * 0.7 + box * 0.3
img[0:95, 280:500] = box2
# 图像可视化
cv.imshow('image', img)
# 图像暂停时间
cv.waitKey(0)
# 释放所有资源
cv.destroyAllWindows()

# 图像通道的分割和合并
b, g, r = cv.split(img)
img = cv.merge((r, g, b))
# 图像可视化
cv.imshow('image', img)
# 停留
cv.waitKey(0)
# 释放所有资源
cv.destroyAllWindows()

# 图像合并
img1 = cv.imread('t1.png')
img2 = cv.imread('t2.png')
# 将图片设置为同样大小
img1 = cv.resize(img1, (300, 300))
img2 = cv.resize(img2, (300, 300))
# 添加背景
" 计算公式: dst = alpha * src1 + beta * rc2 + gamma "
dst = cv.addWeighted(src1=img1, alpha=136, src2=img2, beta=1.0, gamma=0)
# 图像可视化
cv.imshow('image', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 添加边框
img = cv.imread('opencv-logo.png')
'添加边框'
# 直接复制
replicate = cv.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_REPLICATE)
# 边界反射
reflect = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT)
# 边界反射, 边界像素不保留
reflect101 = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
# 边界延伸
wrap = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_WRAP)
# 添加常数
constant = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_WRAP, value=[128, 1288, 128])
# 可视化
plt.subplot(231)
plt.imshow(cv.cvtColor(img, cv.COLOR_BAYER_BG2BGR))
plt.title('Original')
plt.subplot(232)
plt.imshow(cv.cvtColor(replicate, cv.COLOR_BAYER_BG2BGR))
plt.title('replicate')
plt.show()
plt.subplot(233)
plt.imshow(cv.cvtColor(reflect, cv.COLOR_BAYER_BG2BGR))
plt.title('reflectl')
plt.show()
plt.subplot(234)
plt.imshow(cv.cvtColor(reflect101, cv.COLOR_BAYER_BG2BGR))
plt.title('reflect101')
plt.subplot(235)
plt.imshow(cv.cvtColor(wrap, cv.COLOR_BAYER_BG2BGR))
plt.title('wrap')
plt.subplot(236)
plt.imshow(cv.cvtColor(constant, cv.COLOR_BAYER_BG2BGR))
plt.title('constant')
plt.show()

img = np.array(np.arange(0, 54)).reshape((3, 3, 6))
print(img)
img2 = cv.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_REPLICATE)
print(img2.shape)
print(np.transpose(img, (2, 0 ,1)))
print(np.transpose(img2, (2, 0 ,1)))

# 图像位运算(将logo放在图像上角)
img1 = cv.imread('t1.png')
img2 = cv.imread('t2.png')
# 获取一个新数据(右上角数据)
rowsl, colsl, _ = img1.shape
rows, cols, channels = img2.shape
start_rows = 0
end_rows = rows
start_cols = colsl - cols       # 在这里改位置
end_cols = colsl
roi = imgl[start_rows: end_rows, start_cols: end_cols]
# 将图像转换为灰度图
img2gray = cv.cvtColor(img2, cv.COLOR_BAYER_BG2BGR)
# 将灰度图转换为黑白图
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# 求反操作, 即255-mask
mask_inv = cv.bitwise_not(mask)
# 获取得到背景图(对应mask_inv为True的时候, 进行and操作, 其他位置为0)
# 求解bitwise_and操作的时候, 如果mask给定, 只对mask中对应为1的位置进行and操作, 其他位置不变
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# 获取得到前景图
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
# 前景颜色和背景颜色合并
dst = cv.add(img1_bg, img2_fg)
# 复制粘贴
img1[start_rows: end_rows, start_cols: end_cols] = img2
# 可视化
cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()

# 图像基本处理
'颜色提取'
img = cv.imread('opencv-logo.png')
# 转化内HSV格式
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# 定义像素点范围
'蓝色范围'
lower = np.array([150, 50, 50])
upper = np.array([130, 255, 255])
'红色范围'
lower = np.array([150, 50, 50])
upper = np.array([200, 255, 255])
# 在这个范围的图像像素设置为255, 不为0
mask = cv.inRange(hsv, lower, upper)
# 进行And操作进行数据合并
dst = cv.bitwise_and(img, img, mask=mask)
# 图像可视化
cv.imshow('hsv', hsv)
cv.imshow('mask', mask)
cv.imshow('image', img)
cv.imshow('dest', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 图像大小改变
img = cv.imread('t1.png')
old_height, old_width, _  = img.shape
print(f'旧图像的大小, 高度={old_height}, 宽度:{old_width}')
new_height = int(0.8 * old_height)
new_width = 250
print(f'新图像的大小, 高度={new_height}, 宽度:{new_width}')
dst = cv.resize(img, (new_width, new_height))
print(dst.shape)
# 图像可视化
cv.imshow('image',dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 图像平移
img = cv.imread('t1.png')
# 构建一个M
M = np.float32([
    [1, 0, -10],
    [0, 1, -100]
])
# warpAffine计算规则: src(x, y) = dst(m11*x + m12*y + m13, m21*x + m22*y + m23)  其中x y是坐标
dst = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
# 可视化
cv.imshow('dest', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 图像旋转
img = cv.imread('t1.png')
rows, cols, _ = img.shape
'构建一个旋转中心点, 旋转大小, 尺度, angle: 负数表示逆时针旋转'
M = cv.getRotationMatrix2D(CENTER=(cols/2, rows/2), angle=20, scale=1)
dst = cv.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
print(dst.shape)
# 可视化
cv.imshow('imag', dst)
cv.waitKey(0)
cv.destroyAllWindows()
# 90度, 顺时针
dst1 = cv.rotate(img, rotateCode=cv.ROTATE_90_CLOCKWISE)
print(dst1.shape)
# 可视化
cv.imshow('dst1', dst1)
cv.waitKey(0)
cv.destroyAllWindows()

# 水平或者垂直翻转
dst2 = cv.flip(img, 0)    # 上下翻转
dst3 = cv.flip(img, 1)    # 左右翻转
print(dst2.shape)
print(dst3.shape)
# 可视化
cv.imshow('dst2', dst2)
cv.imshow('dst3', dst3)
cv.waitKey(0)
cv.destroyAllWindows()
'实例'
img = np.array(range(9), shape=(3, 3))
print(img)
cv.flip(img, 1)
print(img[:, ::-1])
'可实际操作'
import cv2 as cv 
a = cv.imread('./kb.png')
dst = cv.flip(a, 0)
print(dst.shape)
cv.imshow('dst', dst )
cv.waitKey(0)
cv.destroyAllWindows()

# 仿射变换
img = cv.imread('t1.png')
h, w, _ = img.shape
M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
dst1 = cv.warpAffine(img, M, (w+30, w//2), borderValue=[0, 0, 0])
# 分布在原水图像中选择三个点以及在这三个点在新图像中位置
pts1 = np.float32([[170, 200], [350, 200], [170, 400]])
pts2 = np.float32([[10, 50], [200, 50], [100, 300]])
# 构建对应M
M = cv.getAffineTransform(pts1, pts2)
print(M)
# 进行转换
dst = cv.warpAffine(img, M, (h, w))
# 可视化
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Input')
plt.subplot(122)
plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
plt.title('Output')
plt.subplot(123)
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.title('dst')

# 图像旋转水平
img = cv.imread('t1.png')
h, w, _ = img.shape
M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
dst = cv.warpAffine(img, M, (w+30, w//2), borderValue=[0, 0, 0])
cv.imshow('mask1', img)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 透视转换
'根据四个点来进行转换操作, 形状不会变化, 任意三个点均不在同一线上'
img = cv.imread('t1.png')
rows, cols, _ = img.shape
# 画两条线
cv.line(img, pt1=(0, rows//2), pt2=(cols, rows//2), color=(255, 0, 0), thickness=5)
cv.line(img, pt1=(cols//2, 0), pt2=(cols//2, rows), color=(255, 0, 0), thickness=5)
# 定义四个点
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
# m是一个3*3矩阵
m = cv.getPerspectiveTransform(pts1, pts2)
print(m)
# 计算规则: dst(x, y) = src((m11x+m12y+m13)/(m31x+m32y+m33)), (m21x+m22y+m23) / (m31x+m32y+m33)
dst = cv.warpPerspective(img, m, (300, 300))
# 可视化
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Input')
plt.subplot(122)
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.title('Output')

# 二值化图像
'产生一个图像(偶从白->黑)'
img = np.arange(2555, 1, -1).reshape((1, -1))
for i in range(255):
    img = np.append(img, np.arange(255, -1, -1).reshape((1, -1)), axis=0)
img = img.astype(np.uint8)
'进行普通二值化操作(参数1->返回的阈值, 参数2->返回的是二值化后的图像)'
ret, thresh1 = cv.threshold(src=img, thresh=27, maxval=255, type=cv.THRESH_BINARY)
'反转的二值化操作, 将小于等于玉坠thresh的设置为maxval, 大于该值的设置为0'
ret, thresh2 = cv.threshold(src=img, thresh=27, maxval=255, type=cv.THRESH_BINARY_INV)
'截断二值化操作, 将小于等于阈值thresh的设置为原始值, 大于该值的设置为maxval'
ret, thresh3 = cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_TRUNC)
'0二值化操作, 将小于等于玉坠thresh的设置为0, 大于该值设置为原始值'
ret, thresh4 = cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_TOZERO)
'反转0二值化操作, 将小于等于玉坠thresh的设置为原始值, 大于该值设置为0'
ret, thresh5 = cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_TOZERO_INV)

titles = ['Origina Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 进行自适应二值化操作
'二值化操作的时候需要给定一个阈值, 实际情况阈值不好计算, 所以可以基于图像本身, 根据母亲区域像素值获取适合阈值对当前进行二值化操作'
img = cv.imread('t1.png', 0)
'普通二值化操作'
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
'使用均值方式产生当前像素点对应阈值, 使用(x, y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值'
th2 = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, 
                            thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
'使用高斯分布产生当前像素点对应阈值, 使用(x, y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值'
'其中权重为和当前数据有关的搞事随机数'
th3 = cv.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
titles = ['Origina Image', 'Global Thresholding(v=127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 灰度转换(主要和绿色有关)
Gray = img[:, :, 1]*0.59
Gray
# 边界填充
cv.copyMakeBorder()

# 图像二值化 + 高斯模糊
'产生噪声数据'
img1 = np.random.normal(150, 100, size=(300, 300))
img1 = np.clip(img1, 0, 255)
img1 = img1.astype(np.uint8)
'产生背景图像'
img2 = np.zeros((300, 300), dtype=np.uint8)
img2[100: 200, 100: 200] = 255
'合并两张图像, 得到一张图像'
img = cv.addWeighted(src1=img1, alpha=0.3, src2=img2, beta=0.3, gamma=0)

# 进行大津法二值化操作(找一个最大基于直方图的最大差异性阈值点)
'进行普通二值化操作'
ret1, th1 = cv.threshold(img, 27, 255, cv.THRESH_BINARY)
'进行大津法二值化操作'
tet2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 图像卷积操作
img1 = cv.imread('t1.png')
'自定义一个kernel核'
kernel = np.ones((3, 3), np.float32) / 9
'做一个卷积操作'
img2 = cv.filter2D(img1, -1, kernel)    # 第二个参数为: depth 一般为-1
'做大小缩放'
h, w, _ = img1.shape
w = int(w*0.5)
h = int(h*0.5)
img3 = cv.resize(img1, (w, h))
img4 = cv.resize(img2, (w, h))
w = int(w*0.2)
h = int(h*0.2)
img5 = cv.resize(img3, (w, h))
img6 = cv.resize(img4, (w, h))
dst = cv.filter2D(img6, -1, kernel)
'可视化'
plt.subplot(121)
plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.show()
plt.subplot(122)
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
plt.title('Define Kernal')
plt.show()


# 均值滤波
img = cv.imread('t1.png')
'做一个卷积操作'
dst = cv.blur(img, ksize=(11, 11))
'可视化'
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')

# 高斯滤波
a = cv.getGaussianKernel(5, 1, ktype=cv.CV_64F)
print(a)
kernel = np.dot(a, a.T)
print(kernel)

# 定义一个卷积核
kernel1 = cv.getGaussianKernel(9, 2, ktype=cv.CV_64F)
plt.figure(figsize=(20, 10))
# 可视化
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.show()

# 灰度图转换
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
'做一个中值过滤'
dst = cv.medianBlur(img, ksize=5)
'可视化'
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.show()

# 双边滤波
'中间的纹理删除, 保留边缘信息'
'加载图像'
img = cv.imread('t1.png')
'做一个双边滤波'
dst = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
'可视化'
plt.subplot(121)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.show()

# 图像形态学变换操作
"""
主要有: 
    腐蚀(Erode), 膨胀(Dilate), Open, Close, Morphological Grafient, Top Hat, Black Hat
"""
kernel1 = cv.getStructuringElement(cv.MORPH_RECT, ksize=(5, 5))
print(f'矩形kernel: \n{kernel1}')
kernel2 = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(5, 5))
print(f'十字架kernel: \n{kernel2}')
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
print(f'椭圆kernel: \n{kernel3}')

# 进行腐蚀操作
'1. 定义一个核(全部设置为1表示对核5*5区域的所有像素进行考虑, 设置为0表示不考虑)'
kener = np.ones((5, 5), np.uint8)
'2. 腐蚀操作'
dst = cv.erode(img, kernel, iteration=1, borderType=cv.BORDER_REFLECT)
'3. 可视化'
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Open
'先做一次腐蚀, 然后做一次扩张'
'1. 定义一个核'
kerne = np.ones((5, 5), np.uint8)
'2. 腐蚀'
dst1 = cv.erode(img, kernel, iterations=1)
'3. 膨胀'
dst2 = cv.dilate(dst1, kernel, iterations=1)
'4. 可视化'
cv.imshow('img', img)
cv.imshow('dst1', dst1)
cv.imshow('dst2', dst2)
cv.waitKey(0)
cv.destroyAllWindows()

# Closing
img = cv.imread('t2.png', 0)
'加载噪声数据'
rows, cols = img.shape
'加白色'
for i in range(100):
    x = np.random.randint(cols)
    y = np.random.randint(rows)
    img[y, x] = 255
'加黑色'
for i in range(1000):
    x = np.random.randint(cols)
    y = np.random.randint(rows)
    img[y, x] = 0

# 混合
'先open再close'
dst1 = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel, iterations=11)
dst2 = cv.morphologyEx(dst1, op=cv.MORPH_CLOSE, kernel=kernel, iterations=11)
'可视化'
cv.imshow('img', img)
cv.imshow('dst1', dst1)
cv.imshow('dst2', dst2)
cv.waitKey(0)
cv.destroyAllWindows()

# 形态梯度
'1. 定义一个核'
kernel = np.ones((5, 5), np.uint8)
'2. 形态梯度'
dst = cv.morphologyEx(img, op=cv.MORPH_GRADIENT, kernel=kernel, iterations=1)
'3. 可视化'
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Top Hat/Black Hat
'1. 定义一个核'
kernel = np.ones((5, 5), np.uint8)
'2. close后Black Hat'
dst1 = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)
dst2 = cv.morphologyEx(img, op=cv.MORPH_BLACKHAT, kernel=kernel, iterations=1)
'3. 可视化'
plt.subplot(131)
plt.imshow('img', img)
plt.title('img')
plt.subplot(132)
plt.imshow('dst', dst1)
plt.title('dst')
plt.subplot(133)
plt.imshow('dst', dst2)
plt.title('dst2')
plt.show()

# 梯度Sobel滤波/卷积
'''
梯度和高斯平滑结合, 求梯度前, 需要进行一个高斯平滑
    SobelX: 水平梯度 X 垂直边缘----> 提取垂直边缘信息(矩阵乘法)
    SobelY: 垂直梯度 X 水平边缘----> 提取水平边缘信息(矩阵乘法)
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('t1.png', 0)
'画几条线'
rows, cols = img.shape
cv.line(img, pt1=(0, rows//3), pt2=(cols, rows//3), color=0, thickness=5)
cv.line(img, pt1=(0, 2*rows//3), pt2=(cols, 2*rows//3), color=0, thickness=5)
cv.line(img, pt1=(cols//3), pt2=(cols//3, rows), color=0, thickness=5)
cv.line(img, pt1=(2*cols//3), pt2=(2*cols//3, rows), color=0, thickness=5)
cv.line(img, pt1=(0, 0), pt2=(cols, rows), color=0, thickness=1)

sobelx = cv.Sobel(img, 6, dx=1, dy=0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, dx=0, dy=1, ksize=5)
sobelx2 = cv.Sobel(img, cv.CV_64F, dx=2, dy=0, ksize=5)
sobely2 = cv.Sobel(img, cv.CV_64F, dx=0, dy=2, ksize=5)
sobel = cv.Sobel(img, cv.CV_64F, dx=1, dy=1, ksize=5)
sobel_x = cv.Sobel(img, cv.CV_64F, dx=0, dy=1, ksize=5)
sobel_y = cv.Sobel(img, cv.CV_64F, dx=1, dy=0, ksize=5)
'基于定义一个kernel核'
kerenl = np.asarray([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
kernel1 = np.asarray([[1, 2, 1]])
kernel2 = np.asarray([[-1], [-2], [-1]])
'做一个卷积操作--先高斯平滑'
a = cv.filter2D(img, 6, kernel1.T)
sobelx = cv.filter2D(a, 6, kernel2.T)
'再垂直梯度'
sobely = cv.filter2D(a, 6, kernel2)
'可视化'
cv.imshow('sobely', sobely)
cv.waitKey(0)
cv.destroyAllWindows()

# 梯度Laplacian滤波/卷积
img = cv.imread('t1.png', 0)
'画几条线'
rows, cols = img.shape
cv.line(img, pt1=(0, rows//3), pt2=(cols, rows//3), color=0, thickness=5)
cv.line(img, pt1=(0, 2*rows//3), pt2=(cols, 2*rows//3), color=0, thickness=5)
cv.line(img, pt1=(cols//3), pt2=(cols//3, rows), color=0, thickness=5)
cv.line(img, pt1=(2*cols//3), pt2=(2*cols//3, rows), color=0, thickness=5)
'ksize设置为3'
ksize = 3
laplacian = cv.Laplacian(img, cv.CV_64F, ksize=ksize)
sobelx = cv.Sobel(img, cv.CV_64F, dx=1, dy=0, ksize=ksize)
sobely = cv.Sobel(img, cv.CV_64F, dx=0, dy=1, ksize=ksize)
scharr_x = cv.Scharr(img, cv.CV_64F, dx=1, dy=0)
scharr_y = cv.Scharr(img, cv.CV_64F, dx=0, dy=1)

plt.figure(figsize=(20, 10))
'可视化'
plt.subplot(231)
plt.imshow(img, 'gray')
plt.title('img')
plt.subplot(232)
plt.imshow(sobelx, 'gray')
plt.title('sobelx')
plt.subplot(233)
plt.imshow(sobely, 'gray')
plt.title('sobely')

'ksize设置为5'
ksize = 5
dst1 = cv.Laplacian(img, cv.CV_8U, ksize=ksize)
dst2 = cv.Laplacian(img, cv.CV_64F, ksize=ksize)
dst3 = np.uint8(np.absolute(dst2))
'可视化'
plt.subplot(221)
plt.imshow(img, 'gray')
plt.title(img)
plt.subplot(222)
plt.imshow(dst1, 'gray')
plt.title(dst1)
plt.subplot(223)
plt.imshow(dst2, 'gray')
plt.title(dst2)
plt.subplot(224)
plt.imshow(dst3, 'gray')
plt.title(dst3)

# Canny边缘检测算法
"""
步骤:
    1. 高斯滤波器, 以平滑图像, 滤除噪声
    2. 计算图像中每个像素点的梯度大小和方向
    3. 应用非极大值抑制
    4. 应用双阈值检测确定真实和潜在边缘
    5. 通过抑制孤立的弱边缘最终完成边缘检测
"""
'1. 高斯去噪声'
blur = cv.GaussianBlur(img, (5, 5), 0)
'2 Canny边缘检测'
edges = cv.Canny(blur, threshold1=50, threshold2=250)
'可视化'
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(blur, cmap='gray')
plt.title('blur')
plt.show()

# 轮廓信息
img = cv.imread('t1.png')
'转换为灰度图'
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
'做一个图像反转 0-> 255, 255->0'
img1 = cv.bitwise_not(img1)
'做一个二值化'
ret, thresh = cv.threshold(img1, 12, 2555, cv.THRESH_BINARY)
"--->发现轮廓信息"
'参数1是原始图像, 参数2是了轮廓检索模型, 参数3是参数近似方法'
'返回值1是修改过图像, 2是参数值轮廓, 3是层次信息; CHAIN_APPROX_SIMPLE一条直线上仅保留端点信息'
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
'在图像中绘制图像'
max_idx = np.argmax([len(t) for t in contours])
img3 = cv.drawContours(img, contours, contourIdx=max_idx, color=(0, 0, 255), thickness=2)
'提取轮廓'
img4 = np.zeros_like(img3)
img4 = cv.drawContours(img4, contours, contourIdx=-1, color=255, thickness=2)
'可视化'
plt.figure(figsize=(20, 10))
plt.subplot(231)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('thresh')
plt.subplot(232)
plt.imshow(thresh, camp='gray')
plt.title('thresh')
print(f'轮廓数目为: {len(contours)}')
"""
第一个值表示当前轮廓的上一个同层级的轮廓下标
第二个值表示当前轮廓的下一个同层级的轮廓下标
第三个值表示当前轮廓的第一个同层级的轮廓下标
第一个值表示当前轮廓的父轮廓下标
"""
hierarchy    # 是一个[1, n, 4]格式, n为轮廓数目 

cnt = cv.imread('t2.png')
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
# 绘制最小矩形
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int64(box)
cv.drawContours(img, [box], 0, (0, 255, 0), 2)
print(rect)

# 绘制旋转最小矩形
img = cv.imread('t2.png')
'旋转'
rect = [(302, 420), (317, 373), 18]
M = cv.getRotationMatrix2D(center=rect[0], anggle=rect[-1], scale=1)
img = cv.warpAffine(img, M, (cols, rows), borderValue=[255, 255, 255])
'将图像转化为灰度图像'
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
'反转'
img1 = cv.bitwise_not(img1)
'二值化'
ret1, th = cv.threshold(img1_bg, 27, 255, cv.THRESH_BINARY)
'可视化'
plt.figure(figsize=(20, 10))
plt.subplot(233)
plt.imshow(th)
plt.title('threshold')
plt.show()

# 直方图绘画---像素值
img = cv.imread('t1.png', 0)
'cv绘画'
hist1 = cv.calcHist([img], channels=[0], mask=None, hsitSize=[256], ranges=[0, 256])
'numpy计算'
hist2, bins = np.histogram(img.ravel(), 4, [0, 256])
'np绘画'
hist3 = np.bincount(img.ravel(), minlength=256)
plt.figure(figsize=(20, 10))
'可视化'
plt.subplot(131)
plt.imshow(img, 'gray')
plt.title('Orgina Image')
plt.show()

# 针对色彩图像计算直方图
img = cv.imread('t1.png')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()

# 针对HSV图像计算直方图
img = cv.imread('t1.png')
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
color = ('r', 'y', 'g')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()

# 加入mask -- 找局部区域
img = cv.imread('t1.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
'加入mask位置信息'
mask = np.zeros(img.shape[: 2], np.uint8)
mask[50: 250, 50: 450] = 255    # 位置信息参数
'构建mask区域图像'
masked_img = cv.bitwise_and(img, img, mask=mask)
'cv绘画'
hist1 = cv.calcHist([img], channels=[0], mask=None, hsitSize=[256], ranges=[0, 256])
hist1 = hist1 / np.sum(hist1)
hist2 = cv.calcHist([masked_img], channels=[0], mask=None, hsitSize=[256], ranges=[0, 256])
hist2 = hist2 / np.sum(hist2)
plt.figure(figsize=(20, 10))
'可视化'
plt.subplot(131)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Originall Image')
plt.subplot(132)
plt.imshow(cv.cvtColor(hist2, cv.COLOR_BGR2RGB))
plt.title('masked_img')
plt.show()

# 直方图均衡化
img2 = cv.equalizeHist(img)
'均衡'
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img3 = clahe.apply(img)

# 模版匹配
plt.figure(figsize=(16, 16))
img = cv.imread('t1', 0)
'图像copy'
img2 = img.copy()
'加载模版'
template = cv.imread('template.png', 0)
'模版'
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED']
for idx, meth in enumerate(methods):
    method = eval(meth)
    res = cv.matchTemplate(img, template, method)
    '获取位置'
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

# HOG特征
"""
记录梯度 
步骤:
    1. 灰度
    2. 归一化--->调节对比度
    3. 计算像素梯度, 获取轮廓信息
    4. 图像划分为cells
    5. 统计每个cell-->直方图
    6. cell组合成bock
    7. 特征串联, 得到特征向量
"""

# LBP特征
"""
相邻中心像素为阈值, 将相邻8个像素进行比较, 若周围像素值大于中心像素值, 则为1 否则为0                                             
"""

# Haar特征
"""
边缘/ 线性/ 中心  动态监测----级联匹配多个弱分类器变为强分类器
"""
s = "aacabdkacaa"
res = ''
if s == s[::-1]:
    print(s)
res= s[0]

a = ''
for count in range(2, len(s)):
    for i in range(len(s)-count+1):
        a = s[i:i+count]
        if a == a[::-1]:
            res = a
            count += 1
            break
                        
print(res)

# 定位
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
face_cascade = cv.CascadeClassifier('Haarcaascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('Haarcaascade_eye.xml')
# 基于Haar Cascades的Face Detection
img = cv.imread('t1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMulltiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    # 画人脸区域
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 获得人脸区域
    roi_gray = gray[y:y+h, x:x+w]
    roi_img = img[y:y+h, x:x+w]
    # 检测眼睛
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_img, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
# 可视化
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# 从摄像头获取视频 + 人脸区域
# 创建一个摄像头读取, 给定一个视频设备
capture = cv.VideoCapture(0)
# 设置摄像头参数
success = capture.set(cv.CAP_PROP_FRAME_WIDTH, 880)
if success:
    print('设置成功')
success = capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
if success:
    print('设置高度成功')

# 打印属性
size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)

img = frame
# 做一个人脸检测--转灰度图像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 检测图像
faces = face_cascade.detectMulltiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    # 画人脸区域
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 获得人脸区域
    roi_gray = gray[y:y+h, x:x+w]
    roi_img = img[y:y+h, x:x+w]
# 遍历获取视频中图像
# 读取当前时刻的摄像头捕获图像, 返回为值
success, frame = capture.read()
# 遍历以及等待任意键盘输入
while success and cv.waitKey(1) == -1:
    img = frame
    # 做一个人脸检测--转灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 检测图像
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        # 发送业务服务器处理
        pass
    cv. imshow('frame', img)

# 定义一个图像预处理的相关代码
img = cv.imread('t1.png')
def preprocess(gray):
    """
    对灰度对象进行形态转换
    :param gray:
    :return:
    """
    # 高斯平滑
    gaussian = cv.GaussianBlur(gray, (3, 3), 0, 0, cv.BORDER_DEFAULT)
    # 中值滤波
    median = cv.medianBlur(gaussian, 5)
    # Sobel算子, 对边缘进行处理
    sobel = cv.Sobel(median, cv.CV_64F, dX=1, dy=0, ksize=3)
    # 类型转换unit8
    sobel = np.uint8(np.absolute(sobel))
    # 二值化
    ret, binary = cv.threshold(sobel, 170, 255, cv.THRESH_BINARY)
    # 膨胀&腐蚀
    element1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
    element2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 7))
    # 膨胀一次, 让轮廓突出
    dilation = cv.dilate(binary, element2, iterations=1)
    # 腐蚀一次, 去掉细节
    erosion = cv.erode(dilation, element1, iterations=1)
    # 再次膨胀, 让轮廓明显
    dilation2 = cv.dilate(erosion, element2, iterations=3)
    cv.imshow('gaussi   an', gaussian)
    cv.imshow('dilation2', dilation2)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 车牌区域查找
def find_number(img):
    # 查找轮廓
    contours, hiierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 查找矩形
    max_ratio = -1
    max_box = None
    ratios = []
    number = 0
    for i in range(len(contours)):
        cnt = contours[i]     # 当前轮廓坐标信息
        # 计算轮廓面积
        area = cv.contourArea(cnt)
        # 面积太小过滤
        if area < 10000:
            continue
        # 找到最小矩形
        rect = cv.minAreaRect(cnt)
        # 矩形的四个坐标(顺序不一定, 一定是左下角, 左上角, 右上角, 右下角)
        box = cv.boxPoints(rect)
        # 转换为long类型
        box = np.int64(box)
        # 计算长宽高---计算第一条边长度
        a = abs(box[0][0] - box[1][0])
        b = abs(box[0][1] - box[1][1])
        h = np.sqrt(a ** 2 + b ** 2)
        # 计算长宽高---计算第二条边长度
        c = abs(box[1][0] - box[2][0])
        d = abs(box[1][1] - box[2][1])
        w = np.sqrt(a ** 2 + b ** 2)
        # 让最小值为高度, 最大值为宽度
        height = int(min(h, w))
        weight = int(max(h, w))
        # 计算面积
        area2 = height * weight
        # 两个面积差值一定在一个范围内
        r = np.absolute((area2 - area) / area)
        if r > 0.6:
            continue
        ratio = float(weight) / float(height)
        print((box, height, weight, r, ratio, rect[-1]))
        cv.drawContours(img, [box], 0, 255, 2)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # 根据找到的图像矩阵数量进行数据输出
        number += 1
        ratios.append((box, ratio))
        if number == 1:
            # 直接返回
            return ratios[0][0]
        elif number > 1:
            # 直接获取中间值
            filter_ratios = list(filter(lambda t: t[1] >= 2.7 and t[1] < 5.0, ratios))
            size_filter_ratios = len(filter_ratios)

# 卷积神经网络
"""
有效降低反馈神经网络的复杂性(传统神经网络)
常见结构:
    LeNet-5, AlexNet, ZFNet, VGGNet, GoogleNet, ResNet, DenseNet, SENet, Reidual Attention Networks
    MobileNet, ShffleNet
目的:
    增加非线性神经元得出的近似结构, 同时得出更好的特征表达----同时增加复杂度, 难以优化
应用场景:
    图像分两类, 目标检测, 图像分割
"""

# input layer输入层
"""
常见3种数据预处理方式
去均值:
    将输入数据各个维度中心化到0
PCA(去相关)白化:
    去除相关性
归一化, 标准化:
    缩放0-1
opencv图像特征转换:
    几何变换, 颜色处理, 图像合成, 图像增强
"""

# conv layer
"""
局部关联: 
    每个神经元看做嘤filter/kernal
窗口滑动:
    filter/kernal会对图像各个区域进行局部数据计算
局部感知:
    计算时, 将图片划分为一个个区域进行计算
参数共享机制:
    神经元连接数据窗权重固定
"""

# 池化层
"""
放大缩小, 数据增强---提取特征, 卷积池化
剔除噪声
"""
import cv2 as cv
import torch
import torch.nn as nn
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.n_class = 10
        # self.conv1 = nn.Conv2d(
        #     in_channels=3,   # 输入通道
        #     out_channels=10,     # 输出通道数目
        #     kernel_size=(3, 3),  # 卷积核
        #     stride=(1, 1),    # 窗口滑动步长
        #     padding='valid'    # 填充大小 padding为'same'时,自d填充.'valid'不填充
        # )
        # self.act1 = nn.PReLU(num_parameters=10)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, (3, 3), (1, 1), 1),
            nn.PReLU(20),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)) # 自适应d平均池化, 最终输出一d是7*7dfeature map
        )

        self.classify = nn.Sequential(
            nn.Linear(1470, 512),   # 1470 = 30 * 7 *7
            nn.PReLU(1),
            nn.Linear(512, 128),   # 1470 = 30 * 7 *7
            nn.PReLU(1),
            nn.Linear(128, self.n_class)
        )
        
    def forward(self, x):
        """
        一个普通卷积神经网络d前向执行过程
        假d是一个图像分类d模型
        :param x: [N, C, H, W]
        : RETURN : [N, n_clas]
        """
        # z = self.conv1(x)
        # z = self.act1(z)
        # z = self.pool1(z)
        z = self.features(x)    # 总共有n个图像, 每个图像有30个feature map, 每个feature map7*7

        # 2. 全连接 + 激活 ---> 全特征提取
        # 结构转换[N, 30, 7, 7] ---> [N, 30*7*7]
        z = z.view(z.shape[0], -1)
        return self.classify(z)  

if __name__ == '__main__':
    img = torch.rand(10, 3, 244, 244)
    net = Network()
    r = net(img)
    print(r.shape)

# Batch Normalization Layer
"""
标准化处理---标准正态分布或者scale 0-1分布
尺度偏移
"""
bn = nn.BatchNorm1d(num_features=20, eps=1e-16)
x2 = bn(torch.from_numpy(x).to(torch.float)).detach().numpy()
print('+'*50)
print(x2)

