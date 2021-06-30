import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from PRGui import *

import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2 as cv
from pywt import dwt2, idwt2

### 错误类型
class imageSizeError(Exception):
    def __init__(self):
        self.value = "图片大小错误"
    def __str__(self):
        return self.value

class Algorithmm(Gui,QMainWindow):
    imagePaths = []
    originImages = []
    imageList = []
    hideLayoutTag = -1

    def __init__(self,parent=None):
        super(Gui,self).__init__(parent)
        ### 将initUI当作一个函数调用
        self.initUI(self)
        self.signalSlots()

    def signalSlots(self):
        #文件按钮相关方法
        #打开  使用lambda方法,向函数传入参数
        self.openAct.triggered.connect(lambda : importImage(self))
        # #保存
        # self.saveAct.triggered.connect(lambda : importImage(self))
        #退出
        self.exitAct.triggered.connect(self.close)

        ### 平移变换
        self.MoveAct.triggered.connect(lambda:MoveAct(self))
        ### 尺度变换
        self.ScaleAct.triggered.connect(lambda:ScaleActt(self))
        ### 旋转变换
        self.RotateAct.triggered.connect(lambda:RotateActt(self))
        ### 仿射变换
        self.AffineAct.triggered.connect(lambda:AffineActt(self))

        ### 灰度映射
        self.GrayMenu.triggered.connect(lambda:GrayMenuu(self))
        ### 算术运算
        self.Math.triggered.connect(lambda:Mathh(self))
        ### 直方图修正
        self.Histogram.triggered.connect(lambda:Histogramm(self))
        ### 空域滤波
        self.FilterAct.triggered.connect(lambda:FilterActt(self))

        ###Founrier
        self.Founrier.triggered.connect(lambda:Founrierr(self))
        ### ideal
        self.ideal.triggered.connect(lambda:ideal(self))
        ### Butterworth
        self.butterworth.triggered.connect(lambda:butterworth(self))
        ### exponential
        self.exponential.triggered.connect(lambda:exponentiall(self))
        ### Teshu
        self.TeShu.triggered.connect(lambda:TeShuu(self))
        ### DAITZTONG
        ### TongTai
        self.TongTai.triggered.connect(lambda:TongTai(self))

        ### KongYU
        self.KongYu.triggered.connect(lambda:KongYU(self))
        ### Zuhelubo
        self.Zuhelubo.triggered.connect(lambda:Zuheluboo(self))

        ###位平面编码
        self.WeiPingMian.triggered.connect(lambda:WeiPingMiann(self))

        ###DPCM编码
        self.DPCMCode.triggered.connect(lambda:DPCMCodee(self))
        ###余弦变换编码
        self.Cose.triggered.connect(lambda:Cosee(self))
        ###小波变换编码
        self.XiaoWave.triggered.connect(lambda:XiaoWavee(self))

        ### 动态分割
        self.DyaCut.triggered.connect(lambda:DyaCutt(self))
        ### 单阈值分割
        self.Singal.triggered.connect(lambda:Singall(self))

        ###SUSAN边缘检测
        self.SUSAN.triggered.connect(lambda:SUSANN(self))
        ###分水岭分割
        self.FenWaterCut.triggered.connect(lambda:FenWaterCutt(self))


        ###形态学
        self.Binary.triggered.connect(lambda:Binary(self))
        self.BinaryApplice.triggered.connect(lambda:BinaryApplice(self))
        self.GrayXing.triggered.connect(lambda:GrayXing(self))
        self.GrayXingAp.triggered.connect(lambda:GrayXingAp(self))

        ###算子
        self.Sobel.triggered.connect(lambda:Sobel(self))
        self.Roberts.triggered.connect(lambda:Roberts_filter(self))
        self.LPLS.triggered.connect(lambda:Laplacian_filter(self))
        self.Canny.triggered.connect(lambda:Canny_filter(self))
        self.Prewitt.triggered.connect(lambda:Prewitt_filter(self))
        self.HLPLS.triggered.connect(lambda:LoGaussian_filter(self))


def importImage(window):
    fname, _ = QFileDialog.getOpenFileName(window, 'Open file', '.', 'Image Files(*.jpg *.bmp *.png *.jpeg *.rgb *.tif)')
    if fname!='':
        # window.importImageEdit.setText(fname)
        window.imagePaths = []
        window.originImages = []
        window.imageList = []
        window.imagePaths.append(fname)
    if window.imagePaths!=[]:
        readIamge(window)
        resizeFromList(window, window.originImages)
        showImage(window)

def readIamge(window):
    window.originImages=[]
    for path in window.imagePaths:
        imgs=[]
        # img=cv2.imread(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        imgs.append(img)
        window.originImages.append(imgs)

def resizeFromList(window,imageList):
    width=600
    height=600
    window.imageList=[]
    for x_pos in range(len(imageList)):
        imgs=[]
        for img in imageList[x_pos]:
            # image=cv2.resize(img, (width, height))
            image=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)
        # print(len(window.imageList),len(window.imageList[0]))

def showImage(window,headers=[]):
    window.showImageView.clear()
    window.showImageView.setColumnCount(len(window.imageList[0]))
    window.showImageView.setRowCount(len(window.imageList))

    window.showImageView.setShowGrid(False)
    window.showImageView.setEditTriggers(QAbstractItemView.NoEditTriggers)
    window.showImageView.setHorizontalHeaderLabels(headers)
    for x in range(len(window.imageList[0])):
        for y in range(len(window.imageList)):
            imageView=QGraphicsView()
            imageView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            imageView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            img=window.imageList[y][x]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width=img.shape[1]
            height=img.shape[0]

            window.showImageView.setColumnWidth(x, width)
            window.showImageView.setRowHeight(y, height)

            frame = QImage(img, width, height, QImage.Format_RGB888)
            #调用QPixmap命令，建立一个图像存放框
            pix = QPixmap.fromImage(frame)
            item = QGraphicsPixmapItem(pix) 
            scene = QGraphicsScene()  # 创建场景
            scene.addItem(item)
            imageView.setScene(scene)
            window.showImageView.setCellWidget(y, x, imageView)


### 平移
def MoveAct(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows, cols = img[0].shape[:2]
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        result = cv2.warpAffine(img[0], M, (cols, rows))
        # cv2.imshow("move", result)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '平移后'])

### 尺度变换
def ScaleActt(window):
    # imageList = []
    for img in window.originImages:
        imgs = []
        ### 在X轴和Y轴上进行0.5倍的缩放，采用像素区域关系重新采样的插值方法
        dst = cv.resize(img[0],None,fx=0.5,fy=0.5,interpolation=cv.INTER_CUBIC)
        # imgs.extend([img[0],dst])
        # imageList.append(imgs)
        cv2.imshow("ScaleAct", dst)
    # resizeFromList(window,imageList)
    # showImage(window, ['原图', '尺度变换后'])

def RotateActt(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows, cols = img[0].shape[:2]
        ### 设置旋转中心，旋转角度，以及旋转后的缩放比例
        M = cv.getRotationMatrix2D(((cols-1)/2,(rows-1)/2),45,1)
        ### M为仿射变换矩阵，第三个参数是变化后的大小，第四个参数为边界外填充颜色
        dst = cv.warpAffine(img[0],M,(rows,cols),borderValue=(255,255,255))
        imgs.extend([img[0],dst])
        imageList.append(imgs)
        cv.imshow("Rotate",dst)
    resizeFromList(window,imageList)
    showImage(window, ['原图', '旋转变换后'])

def AffineActt(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows,cols = img[0].shape[:2]
        ### 设置三个变换前后的的位置对应点
        pos1 = np.float32([[50,50],[300,50],[50,200]])
        pos2 = np.float32([[10,100],[200,50],[100,250]])
        ### 设置变换矩阵M,获得仿射矩阵
        M = cv.getAffineTransform(pos1,pos2)
        ### warpAffine函数会对M函数进行求解，第三个参数是变换后的大小
        dst = cv.warpAffine(img[0],M,(rows,cols))
        imgs.extend([img[0],dst])
        imageList.append(imgs)
        cv.imshow("AffineAct",dst)
    resizeFromList(window,imageList)
    showImage(window, ['原图', '仿射变换后'])

def GrayMenuu(window):

    for img in window.originImages:
        graypic = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        ###取反
        rev_img = 255 - np.array(graypic)
        
        ### 动态范围压缩
        log_img = np.uint8(42 * np.log(1.0+graypic))

        ### 阶梯量化，对于不满足灰度值范围的像素点进行压缩
        step_img = np.zeros((graypic.shape[0], graypic.shape[1]))
        for i in range(graypic.shape[0]):
            for j in range(graypic.shape[1]):
                if (graypic[i, j] <= 230) and (graypic[i, j] >= 120):
                    step_img[i, j] = 0
                else:
            	    step_img[i, j] = graypic[i, j]

        ### 进行自适应阈值处理，第二个参数当满足条件的像素点被设置的灰度值,第三个参数是自适应阈值算法
        ### 第四个参数是二值化的方法，11是分成的区域大小，2是偏移参数
        ### 阈值分割函数
        threshold_img = cv.adaptiveThreshold(graypic, 254,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

        imagehh = np.hstack([step_img,rev_img])
        imagevv = np.hstack([log_img,threshold_img])
        allimage = np.vstack([imagehh,imagevv])
        cv.namedWindow("GrayMenu", cv.WINDOW_NORMAL)
        cv.imshow('GrayMenu',allimage)


def Mathh(window):
    for img in window.originImages:
        ### 图像相加
        add_img = cv.add(img[0],img[0])

        ### 平均值去噪
        img_medianBlur = cv.medianBlur(img[0],3) #中值滤波

        ### 图像相减
        sub_img = img[0] - img[0]

        imagehh = np.hstack([img[0],add_img])
        imagevv = np.hstack([img_medianBlur,sub_img])
        allimage = np.vstack([imagehh,imagevv])
        cv.namedWindow("Mathh", cv.WINDOW_NORMAL)
        cv.imshow('Mathh',allimage)

def Histogramm(window):
    for img in window.originImages:
        graypic = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        ### 直方图均衡化，进行灰度图的像素强度分布拉伸,它的输入只能是灰度图像
        equ_img = cv.equalizeHist(graypic)

        ### 直方图规定化
        hist = np.zeros_like(graypic)
        _, colorChannel = graypic.shape

    for i in range(colorChannel):
        hist_img, _ = np.histogram(graypic[:, i], 256) # get the histogram
        hist_ref, _ = np.histogram(graypic[:, i], 256)
        cdf_img = np.cumsum(hist_img) # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)
 
        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp)) # find the smallest number in tmp, getthe index of this number
            hist[:, i][graypic[:, i] == j] = idx
    
    imagehh = np.hstack([graypic,equ_img,hist])
    cv.namedWindow('Histogram',cv.WINDOW_NORMAL)
    cv.imshow('Histogram',imagehh)

def FilterActt(window):
    for img in window.originImages:
        ### 线性平滑滤波，均值滤波器
        ls_img =cv.blur(img[0], (7, 7))
    
        ### 线性锐化滤波
        kernel_sharpen_1 = np.array([
         [-1, -1, -1],
         [-1, 9, -1],
          [-1, -1, -1]])
        lr_img = cv.filter2D(img[0], -1, kernel_sharpen_1)

        ### 非线性平滑滤波,中值滤波
        nls_img = cv.medianBlur(img[0], 5)

        ### 非线性锐化滤波，双边滤波
        nlr_img = cv.bilateralFilter(img[0], 5, 31, 31)

        imagehh = np.hstack([ls_img,lr_img])
        imagevv = np.hstack([nls_img,nlr_img])
        allimage = np.vstack([imagehh,imagevv])
        cv.namedWindow("FilterAct", cv.WINDOW_NORMAL)
        cv.imshow('FilterAct',allimage)

def Founrierr(window):
    for img in window.originImages:
        graypic = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
    
         ### 二维傅里叶变换，进行频率转换
        fo = np.fft.fft2(graypic)
    
        ### 零频率分量(DC分量)位于左上角，如果要使其居中，则使用fftshift将结果进行偏移
        fshift = np.fft.fftshift(fo)
        fo_img = np.log(np.abs(fshift))
    
        ### 傅里叶逆变换,iffshift反向位移，使(DC分量)再次出现在左上角
        f1shift = np.fft.ifftshift(fshift)
        nfo_img = np.fft.ifft2(f1shift)
        nfo_img = np.abs(nfo_img)

        imagehh = np.hstack([fo_img,nfo_img])
        cv.namedWindow("Founrierr", cv.WINDOW_NORMAL)
        cv.imshow('Founrierr',imagehh)

def ideal(window):
    D0=30
    W=0
    N=2

    for img in window.originImages:
        graypicq = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        ### 离散傅里叶变换
        dft = cv.dft(np.float32(graypicq),flags=cv.DFT_COMPLEX_OUTPUT)
        ### 中心化
        dtf_shift=np.fft.fftshift(dft) 

        rows,cols = img[0].shape[:2]
        crow,ccol = rows // 2,cols // 2 ### 计算频谱中心

        mask=np.ones((rows,cols,2)) #生成rows行cols列的2维矩阵
        maskk=np.ones((rows,cols,2)) #生成rows行cols列的2维矩阵

        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i-crow)**2+(j-ccol)**2)
                if(D > D0):
                    mask[i,j] = 0
                if(D <D0):
                    maskk[i,j]

        lfshift = dtf_shift*mask
        hfshift = dtf_shift*maskk

        # imagehh = np.hstack([lfshift,hfshift])

        cv.namedWindow("ideal1", cv.WINDOW_NORMAL)
        cv.imshow('ideal1',lfshift)

        cv.namedWindow("ideal2", cv.WINDOW_NORMAL)
        cv.imshow('ideal2',hfshift)


def butterworth(window):
    D0=30
    W=0
    N=2

    for img in window.originImages:
        graypicw = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        ### 离散傅里叶变换
        dft = cv.dft(np.float32(graypicw),flags=cv.DFT_COMPLEX_OUTPUT)
        ### 中心化
        dtf_shift=np.fft.fftshift(dft) 

        rows,cols = graypicw[0],graypicw[1]
        crow,ccol = rows // 2,cols // 2 ### 计算频谱中心
        mask=np.ones((rows,cols,2)) #生成rows行cols列的2维矩阵
        maskk=np.ones((rows,cols,2)) #生成rows行cols列的2维矩阵

        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i-crow)**2+(j-ccol)**2)
                mask[i, j] = 1/(1+(D/D0)**(2*N))
                maskk[i, j] = 1/(1+(D0/D)**(2*N))
             
        lfshift = dtf_shift*mask
        hfshift = dtf_shift*maskk

        cv.namedWindow("ideal1", cv.WINDOW_NORMAL)
        cv.imshow('ideal1',lfshift)

        cv.namedWindow("ideal2", cv.WINDOW_NORMAL)
        cv.imshow('ideal2',hfshift)

def exponentiall(window):
    D0=30
    W=0
    N=2

    for img in window.originImages:
        graypicw = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        ### 离散傅里叶变换
        dft = cv.dft(np.float32(graypicw),flags=cv.DFT_COMPLEX_OUTPUT)
        ### 中心化
        dtf_shift=np.fft.fftshift(dft) 

        rows,cols = graypicw[0],graypicw[1]
        crow,ccol = rows // 2,cols // 2 ### 计算频谱中心
        mask=np.ones((rows,cols,2)) #生成rows行cols列的2维矩阵
        maskk=np.ones((rows,cols,2)) #生成rows行cols列的2维矩阵

        for i in range(rows):
            for j in range(cols):
                D = np.sqrt((i-crow)**2+(j-ccol)**2)
                mask[i, j] = np.exp(-(D/D0)**(2*N))
                maskk[i, j] = np.exp(-(D0/D)**(2*N))

        lfshift = dtf_shift*mask
        hfshift = dtf_shift*maskk

        cv.namedWindow("ideal1", cv.WINDOW_NORMAL)
        cv.imshow('ideal1',lfshift)

        cv.namedWindow("ideal2", cv.WINDOW_NORMAL)
        cv.imshow('ideal2',hfshift)

### 矩阵乘法
def decreaseArray(image1, image2):
    if image1.shape == image2.shape:
        image = image1.copy()
        for i in range(image1.shape[0]-1):
            for j in range(image1.shape[1]-1):
                image[i][j] = image1[i][j] - image2[i][j]
                j = j+1
            i = i+1
        return image
    else:
        raise imageSizeError()

### 矩阵加法
def increaseArray(image1, image2):
    if image1.shape == image2.shape:
        image = image1.copy()
        for i in range(image1.shape[0]-1):
            for j in range(image1.shape[1]-1):
                image[i][j] = image1[i][j] + image2[i][j]
                j = j+1
            i = i+1
        return image
    else:
        raise imageSizeError()




def TeShuu(window):
    for img in window.originImages:
        imageAver3 = cv.blur(img[0],(3,3))
        upsharpMask = decreaseArray(img[0],imageAver3)
        imageSharp = increaseArray(img[0],upsharpMask)
        
        himage = np.hstack([img[0],imageAver3])
        hhimage = np.hstack([imageSharp,upsharpMask])
        vimage = np.vstack([himage,hhimage])

        cv.namedWindow('Teshu',cv.WINDOW_NORMAL)
        cv.imshow('Teshu',vimage)

def TongTai(window,d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    for img in window.originImages:
        gray = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))

    himage = np.hstack([gray,dst])
    cv.namedWindow('TongTai',cv.WINDOW_NORMAL)
    cv.imshow('TongTai',himage)


def KongYU(window):
    for img in window.originImages:

        source = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        ### 均值滤波
        result = cv.blur(source,(5,5))

        himage = np.hstack([source,result])
        cv.namedWindow('KongYU',cv.WINDOW_NORMAL)
        cv.imshow('KongYU',himage)

def Zuheluboo(window):
    for img in window.originImages:
        image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
    # plt.subplot(221),plt.imshow(image,'gray'),plt.title('原始图像')
    # plt.subplot(222),plt.imshow(s1,'gray'),plt.title('中心频率域')
        himage = np.hstack([image,s1])

        w , h = image.shape
        flt = np.zeros(image.shape)

        rx1 = w / 4
        ry1 = h / 2

        rx2 = w*3/4
        ry2 = h/2;

        r = min(w,h)/6
        for i in range(1,w):
            for j in range(1,h):
                if ((i - rx1)**2 + (j - ry1)**2 >= r**2) and ((i - rx2)**2 + (j - ry2)**2 >= r**2):
                    flt[i,j] = 1

        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*flt)))
        hhimage = np.hstack([flt,new_img])

        vimage = np.vstack([himage,hhimage])
        cv.namedWindow('Zuhe',cv.WINDOW_NORMAL)
        cv.imshow('Zuhe',vimage)

def WuYuanSuu(window):
    for img in window.originImages:
        image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
        ### 傅里叶变换
        dft = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)
        ### 使用shift函数将频率为0的部分，从左上角移动到中心位置
        dftshift = np.fft.fftshift(dft)
        ### 由于输出的频谱结果是一个复数，需要调用cv.magnitude()函数将傅里叶变换的双通达结果转换为0到255的范围。
        res1= 20*np.log(cv.magnitude(dftshift[:,:,0], dftshift[:,:,1]))

        ### 傅里叶逆变换
        ishift = np.fft.ifftshift(dftshift)
        iimg = cv.idft(ishift)
        res2 = cv.magnitude(iimg[:,:,0], iimg[:,:,1])

        himage = np.hstack([image,res1,res2])
        cv.namedWindow('WuZuhe',cv.WINDOW_NORMAL)
        cv.imshow('WuZuhe',himage)


def WeiPingMiann(window):
    for img in window.originImages:
        image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
        rows,cols=image.shape[0],image.shape[1]
        ### 构造提取矩阵
        x=np.zeros((rows,cols,8),dtype=np.uint8)
        for i in range(8):
            x[:,:,i] = 2**i
        w = np.zeros((rows,cols,8),dtype=np.uint8)
        w1 = np.zeros((rows,cols,8),dtype=np.uint8)
        w2 = np.zeros((rows,cols,8),dtype=np.uint8)
        w3 = np.zeros((rows,cols,8),dtype=np.uint8)
        w4 = np.zeros((rows,cols,8),dtype=np.uint8)
        w5 = np.zeros((rows,cols,8),dtype=np.uint8)
        w6 = np.zeros((rows,cols,8),dtype=np.uint8)
        w7 = np.zeros((rows,cols,8),dtype=np.uint8)
        w8 = np.zeros((rows,cols,8),dtype=np.uint8)
        wp = [w1,w2,w3,w4,w5,w6,w7,w8]
    
        for i in range(8):
            w[:,:,i] = cv2.bitwise_and(image, x[:,:,i]) # 提取位平面
            mask = w[:,:,i]>0                          # 阈值处理
            w[mask] = 255
            wp[i]=w[:,:,i].copy()


        himage = np.hstack(wp)
        cv.namedWindow('WeiPingMiann',cv.WINDOW_NORMAL)
        cv.imshow('WeiPingMiann',himage)

def DPCMCodee(window):
    for img in window.originImages:
        grayimg = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        rows = grayimg.shape[0]
        cols = grayimg.shape[1]

        image1 = grayimg.flatten() #把灰度化后的二维图像降维成一维列表

        for i in range(len(image1)):
            if image1[i] >= 127:
                image1[i] = 255
            if image1[i] < 127:
                image1[i] = 0
        data = []
        image3 = []
        count = 1

        for i in range(len(image1)-1):
            if (count == 1):
                image3.append(image1[i])
            if image1[i] == image1[i+1]:
                count = count + 1
                if i == len(image1) - 2:
                    image3.append(image1[i])
                    data.append(count)
            else:
                data.append(count)
                count = 1
        if(image1[len(image1)-1] != image1[-1]):
            image3.append(image1[len(image1)-1])
            data.append(1)

        ### 压缩率
        ys_rate = len(image3)/len(image1)*100

        ### 行程编码解码
        rec_image = []
        for i in range(len(data)):
            for j in range(data[i]):
                rec_image.append(image3[i])
        rec_image = np.reshape(rec_image,(rows,cols))

        himage = np.hstack([grayimg,rec_image])
        cv.namedWindow('DPCM',cv.WINDOW_NORMAL)
        cv.imshow('DPCM',himage)

def Cosee(window):
    for img in window.originImages:
        image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

        img_dct = cv.dct(np.array(image, np.float32))

        img_dct[0:100, 0:100] = 0

        img_idct = np.array(cv.idct(img_dct), np.uint8)

        dct_out = img_idct

        himage = np.hstack([image,dct_out])
        cv.namedWindow('Cose',cv.WINDOW_NORMAL)
        cv.imshow('Cose',himage)

def tool_Denoising(inputGrayPic,value):    
    result = inputGrayPic    
    height = result.shape[0]    
    weight = result.shape[1]    
    for row in range(height):    
        for col in range(weight):          
            if (abs(result[row, col]) > value):    
                result[row, col] = 0#频率的数值0为低频    
    return result  


def XiaoWavee(window):
    for img in window.originImages:
        image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
        #cA，cH,cV,cD 分别为近似分量(低频分量)、水平细节分量、垂直细节分量和对角细节分量 
        cA,(cH,cV,cD) = dwt2(image,'haar')#dwt2函数第二个参数指定小波基 
        #设置去噪阈值。因为噪音一般都是高频信息，遍历像素，将VALUE的像素点置0    
        VALUE = 60

        #处理水平高频
        cH = tool_Denoising(cH,VALUE)
        #处理垂直高频
        cV = tool_Denoising(cV,VALUE)
        #处理对角线高频
        cD = tool_Denoising(cD,VALUE)
        ### 重构图像
        rebuild = idwt2((cA,(cH,cV,cD)), 'haar')

        himage = np.hstack([cA,cH])
        hhimage = np.hstack([cV,cD])
        vimage = np.vstack([himage,hhimage])

        cv.namedWindow('CAHVD',cv.WINDOW_NORMAL)
        cv.imshow('CAHVD',vimage)

        cv.namedWindow('XiaoWavee',cv.WINDOW_NORMAL)
        cv.imshow('XiaoWavee',rebuild)

def DyaCutt(window):
    for img in window.originImages:
        image = img[0]

        fil = np.array([[-1, -1, 0],
                  [-1, 0, 1],
                  [0, 1, 1]])
        res = cv.filter2D(image, -1, fil) 

        himage = np.hstack([image,res])
        cv.namedWindow('DyaCutt',cv.WINDOW_NORMAL)
        cv.imshow('DyaCutt',himage)

def Singall(window):
    for img in window.originImages:
        image = img[0]
        # Get the height and width of the grayscale image
        rows = image.shape[0]
        cols = image.shape[1]
    
        # Transform grayscale histogram
        grayHist = np.zeros([256], np.uint64) # The gray scale range of the image is 0~255
        for r in range(rows):
            for c in range(cols):
                grayHist[image[r][c]] += 1
                histogram = grayHist
        # Find the gray value corresponding to the maximum peak of the gray histogram
    
        maxLoc = np.where(histogram == np.max(histogram))
        firstPeak = maxLoc[0][0] # 
    
        # Find the gray value corresponding to the second peak of the gray histogram
        measureDists = np.zeros([256], np.float32)
    
        for k in range(256):
            kkk = np.array(k-firstPeak)
            measureDists[k] = pow(kkk, 2) * histogram[k]
    
        maxLoc2 = np.where(measureDists == np.max(measureDists))
        secondPeak = maxLoc2[0][0]
    
        # Find the gray value corresponding to the minimum value between the two peaks as the threshold
    
        if firstPeak > secondPeak: # The first peak is to the right of the second peak
            temp = histogram[int(secondPeak):int(firstPeak)]
            minLoc = np.where(temp == np.min(temp))
            thresh = secondPeak + minLoc[0][0] + 1 # There are multiple troughs, take the trough on the left
        else:
            temp = histogram[int(firstPeak):int(secondPeak)]
            minLoc = np.where(temp == np.min(temp))
            thresh = firstPeak + minLoc[0][0] + 1
            # After finding the threshold, perform threshold processing to obtain a binary image
        threshImage_out = image.copy()
        threshImage_out[threshImage_out > thresh] = 255
        threshImage_out[threshImage_out <= thresh] = 0

        himage = np.hstack([image,threshImage_out])
        cv.namedWindow('Singall',cv.WINDOW_NORMAL)
        cv.imshow('Singall',himage)

### SUSAN边缘检测
def SUSANN(window):
    for img in window.originImages:
        origin = img[0]
        image = img[0].copy()

        ### 阈值
        threshold_value = (int(image.max())-int(image.min()))/10
        offsetX = [
        	-1,0,1,
          -2,-1,0,1,2,
        -3,-2,-1,0,1,2,3,
        -3,-2,-1,0,1,2,3,
        -3,-2,-1,0,1,2,3,
           -2,-1,0,1,2,
              -1,0,1
        ]
        offsetY = [
        	-3,-3,-3,
          -2,-2,-2,-2,-2,
        -1,-1,-1,-1,-1,-1,-1,
        0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,
           2,2,2,2,2,
              3,3,3
        ]
    
        ### 利用圆形模板遍历图像，计算每点处的USAN值
        for i in range(3,image.shape[0]-3):
            for j in range(3,image.shape[1]-3):
                same = 0
                for k in range(0,37):
                    if abs(int(image[i+int(offsetY[k]),j+int(offsetX[k]),0])-int(image[i,j,0]))<threshold_value:
                        same+=1
            if same < 18:
                image[i, j, 0] = 18 - same
                image[i, j, 1] = 18 - same
                image[i, j, 2] = 18 - same
            else:
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 0

        ### X轴偏移
        X = [-1, -1, -1, 0, 0, 1, 1, 1] 
        ### Y轴偏移
        Y = [-1, 0, 1, -1, 1, -1, 0, 1]
        for i in range(4, image.shape[0]-4):
            for j in range(4, image.shape[1]-4):
                flag = 0
                for k in range(0, 8):
                    if image[i, j, 0] <= image[int(i + X[k]), int(j + Y[k]),0]:
                        flag += 1
                        break
                    if flag == 0: # 判断是否是周围8个点中最大的那个值，是的话则保留
                        image[i, j, 0] = 255
                        image[i, j, 1] = 255
                        image[i, j, 2] = 255
                    else:
                        image[i, j, 0] = 0
                        image[i, j, 1] = 0
                        image[i, j, 2] = 0

        himage = np.hstack([origin,image])
        cv.namedWindow('PicCutButtonn',cv.WINDOW_NORMAL)
        cv.imshow('PicCutButtonn',himage)


# def Fitstt(window):
#     for img in window.originImages:

### 分水岭分割
def FenWaterCutt(window):
    for img in window.originImages:
        original = img[0]
        image = original.copy()
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

        ### 阈值分割，将图像分为黑白两个部分
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
        ### 对图像进行开运算，先腐蚀再膨胀
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)

        ### 都开运算的结果进行膨胀，得到大部分都是背景的区域
        sure_bg = cv.dilate(opening,kernel,iterations=3)
    
        ### 通过distanceTransform获取前景区域
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)  # DIST_L1 DIST_C只能对应掩膜为3    DIST_L2 可以为3或者5
        ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

        ### ure_bg与sure_fg相减,得到既有前景又有背景的重合区域   #此区域和轮廓区域的关系未知 
        sure_fg = np.uint8(sure_fg)
        unknow = cv.subtract(sure_bg, sure_fg)
    
        ### 连通区域处理
        ret, markers = cv2.connectedComponents(sure_fg,connectivity=8) #对连通区域进行标号  序号为0- N-1 
        markers = markers + 1           #OpenCV 分水岭算法对物体做的标注必须都大于1，背景为标号为0因此对所有markers加1变成了1-N
        #去掉属于背景区域的部分（即让其变为0，成为背景）
        #返回的是图像矩阵的真值表。
        markers[unknow==255] = 0  

        # Step8.分水岭算法
        watershed = image.copy()
        markers = cv.watershed(watershed, markers)  #分水岭算法后，所有轮廓的像素点被标注为  -1 
    
        watershed[markers == -1] = [255, 0, 0]

        himage = np.hstack([original,watershed])

        cv.namedWindow('FenWaterCutt',cv.WINDOW_NORMAL)
        cv.imshow('FenWaterCutt',himage)



def Binary(window):
    for img in window.originImages:
        image = img[0]
        ### 返回指定形状和尺寸的结构元素

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        ### 膨胀
        dilated = cv.dilate(image,kernel)
        ### 腐蚀
        blur = cv.blur(image,(5,5))
        ### 开操作
        opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        ### 闭操作
        closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

        ###
        himage = np.hstack([dilated,blur])
        hhimage = np.hstack([opening,closing])

        vimage = np.vstack([himage,hhimage])
        cv.namedWindow('Binary',cv.WINDOW_NORMAL)
        cv.imshow('Binary',vimage)

def BinaryApplice(window):
    for img in window.originImages:
        image = img[0]
        ### 去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        noise = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        ### 目标检测
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 14))
        gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)

        copyImg = image.copy()
        rows = image.shape[0]+2
        cols = image.shape[1]+2

        ### 行和列都加2，且为unint8单通道阵列。掩码层，若对整个图像使用，则需要在原行、列上加二。
        mask = np.zeros((rows, cols),np.uint8)
        ### 第三个参数是泛洪算法的种子点，根据该点的像素判断和其相近颜色的像素点
        ### 第四个参数设置颜色，第五个和第六个参数设置边界范围，cv.FLOODFILL_FIXED_RANGE改变图像填充颜色
        cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)

        himage = np.hstack([image,noise])
        hhimage = np.hstack([gradient,copyImg])
        vimage = np.vstack([himage,hhimage])

        cv.namedWindow('BinaryApp',cv.WINDOW_NORMAL)
        cv.imshow('BinaryApp',vimage)

def GrayXing(window):
    for img in window.originImages:
        image = img[0]
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ### 设置核的大小和形状
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 30))
    
        ### 腐蚀与膨胀,iterations设置次数
        eroded = cv.erode(gray,kernel,iterations = 1)  ### 腐蚀
        dilationed = cv.dilate(gray,kernel,iterations = 1) ## 膨胀
    
        ### 开运算和闭运算
        opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)

        himage = np.hstack([eroded,dilationed])
        hhimage = np.hstack([opening,closing])
        vimage = np.vstack([himage,hhimage])

        cv.namedWindow('GrayXing',cv.WINDOW_NORMAL)
        cv.imshow('GrayXing',vimage)


def GrayXingAp(window):
    for img in window.originImages:
        image = img[0]
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 30))

        Gd_out = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
        sm = cv.boxFilter(gray,-1,(3,3),normalize=True)
    
        hat_g_out = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        hat_b_out = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)

        himage = np.hstack([Gd_out,sm])
        hhimage = np.hstack([hat_g_out,hat_b_out])
        vimage = np.vstack([himage,hhimage])

        cv.namedWindow('GrayXingAp',cv.WINDOW_NORMAL)
        cv.imshow('GrayXingAp',vimage)



###
def Sobel(window):
    for img in window.originImages:
        image = img[0]
        x = cv.Sobel(image, cv2.CV_16S, 1, 0)
        y = cv.Sobel(image, cv2.CV_16S, 0, 1)
        absX = cv.convertScaleAbs(x) # Switch back to unit8
        absY = cv.convertScaleAbs(y)
        sobel_out = cv.addWeighted(absX, 0.5, absY, 0.5, 0)


        cv.namedWindow('Sobel',cv.WINDOW_NORMAL)
        cv.imshow('Sobel',sobel_out)

### Roberts
def Roberts_filter(window):
    for img in window.originImages:
        image = img[0]
        # Grayscale processed image
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Roberts
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
        y = cv.filter2D(grayImage, cv.CV_16S, kernely)
        # To uint8, image fusion
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        roberts_out = cv.addWeighted(absX, 0.5, absY, 0.5, 0)


        cv.namedWindow('Roberts_filter',cv.WINDOW_NORMAL)
        cv.imshow('Roberts_filter',roberts_out)

def Laplacian_filter(window):
    for img in window.originImages:
        image = img[0]
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Grayscale color pictures
        dst = cv.Laplacian(grayImage, cv.CV_16S, ksize=3) # Convolution of grayscale images in small steps
        lap_out = cv.convertScaleAbs(dst)

        cv.namedWindow('Laplacian_filter',cv.WINDOW_NORMAL)
        cv.imshow('Laplacian_filter',lap_out)

### Canny
def Canny_filter(window):
    for img in window.originImages:
        image = img[0]
    
        blurred = cv2.GaussianBlur(image, (3, 3), 0) # Gaussian Blur
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) # Gray conversion
        # X Gradient
        xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0) # Compute gradient
        # Y Gradient
        ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
        # edge
        canny_out = cv2.Canny(xgrad, ygrad, 50, 150) # Use high and low thresholds to find image edges

        cv.namedWindow('Canny_filter',cv.WINDOW_NORMAL)
        cv.imshow('Canny_filter',canny_out)

def Prewitt_filter(window):
    for img in window.originImages:
        image = img[0]
    
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Prewitt
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
        x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
        y = cv.filter2D(grayImage, cv.CV_16S, kernely)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        prewitt_out = cv.addWeighted(absX, 0.5, absY, 0.5, 0)


        cv.namedWindow('Prewitt_filter',cv.WINDOW_NORMAL)
        cv.imshow('Prewitt_filter',prewitt_out)

### F.Laplacian of Gaussian
def LoGaussian_filter(window):
    for img in window.originImages:

        image = img[0]
    
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert BGR to gray color space
    
        gaussian = cv.GaussianBlur(grayImage, (3,3), 0)
    
        dst = cv.Laplacian(gaussian, cv.CV_16S, ksize=3)
    
        loG_out = cv.convertScaleAbs(dst)

        cv.namedWindow('LoGaussian_filter',cv.WINDOW_NORMAL)
        cv.imshow('LoGaussian_filter',loG_out)

if __name__=='__main__':
    app = QApplication(sys.argv)
    mv = Algorithmm()
    mv.show()
    sys.exit(app.exec_())