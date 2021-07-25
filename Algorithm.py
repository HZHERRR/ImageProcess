import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PRGui import *
from Mathh import *

import cv2
import numpy as np
import cv2 as cv
from pywt import dwt2, idwt2




class Algorithmm(Gui,QMainWindow):

    def __init__(self,parent=None):
        super(Gui,self).__init__(parent)
        self.imagePaths = []
        self.originImages = []
        self.imageList = []
        self.hideLayoutTag = -1
        ### 将initUI当作一个函数调用
        self.initUI(self)
        self.signalSlots()



    def signalSlots(self):
        #文件按钮相关方法
        #打开  使用lambda方法,向函数传入参数
        self.openAct.triggered.connect(lambda:self.importImage())

        # #保存
        # self.saveAct.triggered.connect(lambda : importImage(self))

        #退出
        self.exitAct.triggered.connect(self.close)

        ### 平移变换
        self.MoveAct.triggered.connect(lambda:self.MoveActt())
        ### 尺度变换
        self.ScaleAct.triggered.connect(lambda:self.ScaleActt())
        ### 旋转变换
        self.RotateAct.triggered.connect(lambda:self.RotateActt())
        ### 仿射变换
        self.AffineAct.triggered.connect(lambda:self.AffineActt())

        ### 灰度映射
        self.GrayMenu.triggered.connect(lambda:self.GrayMenuu())
        ### 算术运算
        self.Math.triggered.connect(lambda:self.Mathh())
        ### 直方图修正
        self.Histogram.triggered.connect(lambda:self.Histogramm())
        ### 空域滤波
        self.FilterAct.triggered.connect(lambda:self.FilterActt())

        ###Founrier
        self.Founrier.triggered.connect(lambda:self.Founrierr())
        ### 高低通滤波器
        self.HighLoww.triggered.connect(lambda:self.HighLow())
        ### 带通带阻滤波器
        self.DTDZ.triggered.connect(lambda:self.BSP())
        ### Teshu
        self.TeShu.triggered.connect(lambda:self.TeShuu())
        ### TongTai
        self.TongTai.triggered.connect(lambda:self.TongTaii())


        ### KongYU
        self.KongYu.triggered.connect(lambda:self.KongYUU())
        ### Zuhelubo
        self.Zuhelubo.triggered.connect(lambda:self.Zuheluboo())
        ### 无约束滤波器
        self.WuYuanSu.triggered.connect(lambda:self.WuYuanSuu())

        ###位平面编码
        self.WeiPingMian.triggered.connect(lambda:self.WeiPingMiann())

        ###DPCM编码
        self.DPCMCode.triggered.connect(lambda:self.DPCMCodee())
        ###余弦变换编码
        self.Cose.triggered.connect(lambda:self.Cosee())
        ###小波变换编码
        self.XiaoWave.triggered.connect(lambda:self.XiaoWavee())

        ### 动态分割
        self.DyaCut.triggered.connect(lambda:self.DyaCutt())
        ### 单阈值分割
        self.Singal.triggered.connect(lambda:self.Singall())

        ###SUSAN边缘检测
        self.SUSAN.triggered.connect(lambda:self.SUSANN())
        ###分水岭分割
        self.FenWaterCut.triggered.connect(lambda:self.FenWaterCutt())


        ###形态学
        self.Binary.triggered.connect(lambda:self.Binaryy())
        self.BinaryApplice.triggered.connect(lambda:self.BinaryApplicee())
        self.GrayXing.triggered.connect(lambda:self.GrayXingg())
        self.GrayXingAp.triggered.connect(lambda:self.GrayXingApp())

        ###算子
        self.Sobel.triggered.connect(lambda:self.Sobell())
        self.Roberts.triggered.connect(lambda:self.Roberts_filterr())
        self.LPLS.triggered.connect(lambda:self.Laplacian_filterr())
        self.Canny.triggered.connect(lambda:self.Canny_filterr())
        self.Prewitt.triggered.connect(lambda:self.Prewitt_filterr())
        self.HLPLS.triggered.connect(lambda:self.LoGaussian_filterr())


    def importImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', 'Image Files(*.jpg *.bmp *.png *.jpeg *.rgb *.tif)')
        if fname!='':
            self.imagePaths = []
            self.originImages = []
            self.imageList = []
            self.imagePaths.append(fname)
        if self.imagePaths!=[]:
            self.readIamge()
            self.resizeFromList(self.originImages)
            self.showImage()

    def readIamge(self):
        self.originImages=[]
        for path in self.imagePaths:
            imgs=[]
            ### 读取具有已知数据类型的二进制数据以及解析简单格式化文本文件的高效方法。
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            imgs.append(img)
            self.originImages.append(imgs)

    def resizeFromList(self,imageList):
        width=800
        height=600
        self.imageList=[]
        for x_pos in range(len(imageList)):
            imgs=[]
            for img in imageList[x_pos]:
                image=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                imgs.append(image)
            self.imageList.append(imgs)


    def showImage(self,headers=[]):
        self.showImageView.clear()
        self.showImageView.setColumnCount(len(self.imageList[0]))
        self.showImageView.setRowCount(len(self.imageList))

        self.showImageView.setShowGrid(False)
        self.showImageView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.showImageView.setHorizontalHeaderLabels(headers)
        for x in range(len(self.imageList[0])):
            for y in range(len(self.imageList)):
                imageView=QGraphicsView()
                imageView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                imageView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

                img=self.imageList[y][x]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                width=img.shape[1]
                height=img.shape[0]

                self.showImageView.setColumnWidth(x, width)
                self.showImageView.setRowHeight(y, height)

                frame = QImage(img, width, height, QImage.Format_RGB888)
                #调用QPixmap命令，建立一个图像存放框
                pix = QPixmap.fromImage(frame)
                item = QGraphicsPixmapItem(pix)
                scene = QGraphicsScene()  # 创建场景
                scene.addItem(item)
                imageView.setScene(scene)
                self.showImageView.setCellWidget(y, x, imageView)

    ### 平移
    def MoveActt(self):
        for img in self.originImages:
            rows, cols = img[0].shape[:2]
            self.MMM = MoveActt(img[0],rows,cols)
            self.MMM.show()

    ### 尺度变换
    def ScaleActt(self):
        for img in self.originImages:
            self.MMM1 = ScaleActt(img[0])
            self.MMM1.show()

    ### 旋转变换
    def RotateActt(self):
        for img in self.originImages:
            rows, cols = img[0].shape[:2]

            ### 设置旋转中心，旋转角度，以及旋转后的缩放比例
            self.MMM2 = RotateActt(img[0],rows,cols)
            self.MMM2.show()

    ### 仿射变换
    def AffineActt(self):
        imageList = []
        for img in self.originImages:
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

    ### 灰度值映射
    def GrayMenuu(self):
        for img in self.originImages:
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

    ### 算术运算
    def Mathh(self):
        for img in self.originImages:
            self.MMM2  = Mathh(img[0])
            self.MMM2.show()

    ### 直方图修正
    def Histogramm(self):
        for img in self.originImages:
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



    ### 空域滤波
    def FilterActt(self):
        for img in self.originImages:
            self.MMM3 = FilterActt(img[0])
            self.MMM3.show()

    ### 傅里叶变换
    def Founrierr(self):
        for img in self.originImages:
            graypic = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
            self.ppp = Fourierrr(graypic)
            self.ppp.initi()


    ### 高低通滤波器
    def HighLow(self):
        for img in self.originImages:
            image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
            self.hhh = KKHighLow(image)
            self.hhh.show()

    ### 带通带阻滤波器
    def BSP(self):
        for img in self.originImages:
            image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
            self.iii = KKBSP(image)
            self.iii.show()


    ### 特殊高通滤波器
    def TeShuu(self):
        for img in self.originImages:
            self.TTT = Teshuu(img[0])
            self.TTT.show()


    ### 同态滤波器
    def TongTaii(self,d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
        for img in self.originImages:
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


    ### 空域滤波器
    def KongYUU(self):
        for img in self.originImages:

            source = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

            ### 均值滤波
            result = cv.blur(source,(5,5))

            himage = np.hstack([source,result])
            cv.namedWindow('KongYU',cv.WINDOW_NORMAL)
            cv.imshow('KongYU',himage)


    ### 均值滤波器
    def Zuheluboo(self):
        for img in self.originImages:
            image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
            self.vvv = Zuhelubooo(image)
            self.vvv.initi()


    ### 无约束滤波器
    def WuYuanSuu(self):
        for img in self.originImages:
            image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
            self.ooo = CWuYueSuuu(image)
            self.ooo.initi()


    ### 位平面编码
    def WeiPingMiann(self):
        for img in self.originImages:
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

    ### DPCMb编码
    def DPCMCodee(self):
        for img in self.originImages:
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

    ###余弦变换
    def Cosee(self):
        for img in self.originImages:
            image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)

            img_dct = cv.dct(np.array(image, np.float32))

            img_dct[0:100, 0:100] = 0

            img_idct = np.array(cv.idct(img_dct), np.uint8)

            dct_out = img_idct

            himage = np.hstack([image,dct_out])
            cv.namedWindow('Cose',cv.WINDOW_NORMAL)
            cv.imshow('Cose',himage)


    def tool_Denoising(self,inputGrayPic,value):    
        result = inputGrayPic    
        height = result.shape[0]    
        weight = result.shape[1]    
        for row in range(height):    
            for col in range(weight):          
                if (abs(result[row, col]) > value):    
                    result[row, col] = 0#频率的数值0为低频    
        return result  

    ###小波变换
    def XiaoWavee(self):
        for img in self.originImages:
            image = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
            #cA，cH,cV,cD 分别为近似分量(低频分量)、水平细节分量、垂直细节分量和对角细节分量 
            cA,(cH,cV,cD) = dwt2(image,'haar')#dwt2函数第二个参数指定小波基 
            #设置去噪阈值。因为噪音一般都是高频信息，遍历像素，将VALUE的像素点置0    
            VALUE = 60

            #处理水平高频
            cH = self.tool_Denoising(cH,VALUE)
            #处理垂直高频
            cV = self.tool_Denoising(cV,VALUE)
            #处理对角线高频
            cD = self.tool_Denoising(cD,VALUE)

            himage = np.hstack([cA,cH])
            hhimage = np.hstack([cV,cD])
            vimage = np.vstack([himage,hhimage])

            cv.namedWindow('AfterProcess',cv.WINDOW_NORMAL)
            cv.imshow('AfterProcess',vimage)



    ### 动态分割
    def DyaCutt(self):
        for img in self.originImages:
            image = img[0]

            fil = np.array([[-1, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 1]])
            res = cv.filter2D(image, -1, fil) 

            cv.namedWindow('DyaCutt',cv.WINDOW_NORMAL)
            cv.imshow('DyaCutt',res)



    ### 单阈值分割
    def Singall(self):
        for img in self.originImages:
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

            cv.namedWindow('Singall',cv.WINDOW_NORMAL)
            cv.imshow('Singall',threshImage_out)




    ### SUSAN边缘检测
    def SUSANN(self):
        for img in self.originImages:
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


            cv.namedWindow('PicCutButtonn',cv.WINDOW_NORMAL)
            cv.imshow('PicCutButtonn',image)




    ### 分水岭分割
    def FenWaterCutt(self):
        for img in self.originImages:
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

            cv.namedWindow('FenWaterCutt',cv.WINDOW_NORMAL)
            cv.imshow('FenWaterCutt',watershed)



    def Binaryy(self):
        for img in self.originImages:
            self.QQQ = KKBinarry(img[0])
            self.QQQ.show()


    def BinaryApplicee(self):
        for img in self.originImages:
            self.QQQ1 = KKBinaryApplicee(img[0])
            self.QQQ1.show()

    def GrayXingg(self):
        for img in self.originImages:
            image = img[0]
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            self.QQQ2 = KKGrayXing(gray)
            self.QQQ2.show()


    def GrayXingApp(self):
        for img in self.originImages:
            image = img[0]
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            self.QQQ3 = KKGrayApp(gray)
            self.QQQ3.show()



    ###Sobel
    def Sobell(window):
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
    def Roberts_filterr(self):
        for img in self.originImages:
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


    def Laplacian_filterr(self):
        for img in self.originImages:
            image = img[0]
            grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Grayscale color pictures
            dst = cv.Laplacian(grayImage, cv.CV_16S, ksize=3) # Convolution of grayscale images in small steps
            lap_out = cv.convertScaleAbs(dst)

            cv.namedWindow('Laplacian_filter',cv.WINDOW_NORMAL)
            cv.imshow('Laplacian_filter',lap_out)

    ### Canny
    def Canny_filterr(self):
        for img in self.originImages:
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



    def Prewitt_filterr(self):
        for img in self.originImages:
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
    def LoGaussian_filterr(self):
        for img in self.originImages:

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