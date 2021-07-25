from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PRGui import *

import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


### 错误类型
class imageSizeError(Exception):
    def __init__(self):
        self.value = "图片大小错误"
    def __str__(self):
        return self.value

class MoveActt(QWidget):
    MoveX=0
    MoveY=0

    def __init__(self,image=None,rows=None,cols=None):
        super().__init__()
        self.initUI()
        self.img = image
        self.rows = rows
        self.cols = cols

    
    def initUI(self):
        self.setWindowTitle('图片平移')
        self.resize(200, 100)

        self.texto = QLabel('请输入X的偏移量',self)
        self.texts = QLabel('请输入Y的偏移量',self)

        self.lineo = QLineEdit(self)
        self.lines = QLineEdit(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.texto)
        layout.addWidget(self.lineo)
        layout.addWidget(self.texts)
        layout.addWidget(self.lines)

        hengzhe = QHBoxLayout(self)
        layout.addLayout(hengzhe)

        self.Buttono = QPushButton('确认',self)
        self.Buttons = QPushButton('取消',self)

        hengzhe.addWidget(self.Buttono)
        hengzhe.addWidget(self.Buttons)

        self.Buttono.clicked.connect(self.actiono)
        self.Buttons.clicked.connect(self.close)

    def actiono(self):
        MoveX = int(self.lineo.text())
        MoveY = int(self.lines.text())
        M = np.float32([[1, 0, MoveX], [0, 1, MoveY]])
        result = cv2.warpAffine(self.img, M, (self.cols, self.rows))
        self.close()
        cv2.imshow("MoveAct",result)



class ScaleActt(QWidget):
    MoveFX=0
    MoveFY=0

    def __init__(self,image=None):
        super().__init__()
        self.initUI()
        self.img = image

    
    def initUI(self):

        self.setWindowTitle('图片缩放')
        self.resize(200, 100)

        self.texto = QLabel('请输入X轴方向的缩放比例',self)
        self.texts = QLabel('请输入Y轴方向的缩放比例',self)

        self.lineo = QLineEdit(self)
        self.lines = QLineEdit(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.texto)
        layout.addWidget(self.lineo)
        layout.addWidget(self.texts)
        layout.addWidget(self.lines)

        hengzhe = QHBoxLayout(self)
        layout.addLayout(hengzhe)

        self.Buttono = QPushButton('确认',self)
        self.Buttons = QPushButton('取消',self)

        hengzhe.addWidget(self.Buttono)
        hengzhe.addWidget(self.Buttons)

        self.Buttono.clicked.connect(self.actiono)
        self.Buttons.clicked.connect(self.close)


    def actiono(self):
        MoveFX = float(self.lineo.text())
        MoveFY = float(self.lines.text())
        result = cv2.resize(self.img,None,fx=MoveFX,fy=MoveFY,interpolation=cv2.INTER_CUBIC)
        self.close()
        cv2.imshow("ScaleAct",result)


class RotateActt(QWidget):
    def __init__(self,image=None,rows=None,cols=None):
        super().__init__()
        self.initUI()
        self.img = image
        self.rows = rows
        self.cols = cols


    def initUI(self):
        self.setWindowTitle('图片缩放')
        self.resize(200, 100)

        self.text = QLabel('请输入旋转的比例',self)
        self.line = QLineEdit(self)

        layout = QVBoxLayout(self)

        layout.addWidget(self.text)
        layout.addWidget(self.line)

        hengzhe = QHBoxLayout(self)
        layout.addLayout(hengzhe)

        self.Buttono = QPushButton('确认',self)
        self.Buttons = QPushButton('取消',self)
        hengzhe.addWidget(self.Buttono)
        hengzhe.addWidget(self.Buttons)

        self.Buttono.clicked.connect(self.actiono)
        self.Buttons.clicked.connect(self.close)

    def actiono(self):
        Rotate = int(self.line.text())
        M = cv2.getRotationMatrix2D(((self.cols-1)/2,(self.rows-1)/2),Rotate,1)
        result = cv2.warpAffine(self.img,M,(self.rows,self.cols),borderValue=(255,255,255))
        self.close()
        cv2.imshow("Rotate",result)


class Mathh(QWidget):

    def __init__(self,image):
        super().__init__()
        self.img = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('算术运算')

        HHH = QHBoxLayout(self)
        VVVo = QVBoxLayout(self)
        VVVs = QVBoxLayout(self)

        HHH.addLayout(VVVo)
        HHH.addLayout(VVVs)

        self.Buttono = QPushButton('图像相加',self)
        self.Buttons = QPushButton('图像相减',self)
        self.Buttont = QPushButton('平均值去噪',self)

        VVVo.addWidget(self.Buttono)
        VVVo.addWidget(self.Buttons)

        VVVs.addWidget(self.Buttont)

        self.Buttono.clicked.connect(self.add)
        self.Buttons.clicked.connect(self.sub)
        self.Buttont.clicked.connect(self.medianBlur)

    def add(self):
        add_img = cv2.add(self.img,self.img)
        cv2.namedWindow('add',cv2.WINDOW_NORMAL)
        cv2.imshow('add',add_img)
        self.close()

    def medianBlur(self):
        img_medianBlur = cv2.medianBlur(self.img,3)
        cv2.namedWindow('median',cv2.WINDOW_NORMAL)
        cv2.imshow('median',img_medianBlur)
        self.close()

    def sub(self):
        sub_img = self.img - self.img
        cv2.namedWindow('sub',cv2.WINDOW_NORMAL)
        cv2.imshow('sub',sub_img)
        self.close()
    
class FilterActt(QWidget):
    def __init__(self,image):
        super().__init__()
        self.img = image
        self.initUI()
    
    def initUI(self):
        HHH = QVBoxLayout(self)
        VVVo = QHBoxLayout(self)
        VVVs = QHBoxLayout(self)

        HHH.addLayout(VVVo)
        HHH.addLayout(VVVs)

        self.Buttono = QPushButton('均值滤波器',self,clicked=self.ls)
        self.Buttons = QPushButton('线性锐化滤波',self,clicked=self.kernel)
        self.Buttont = QPushButton('非线性平滑滤波',self,clicked=self.nls)
        self.Buttonf = QPushButton('非线性锐化滤波',self,clicked=self.nlr)

        VVVo.addWidget(self.Buttono)
        VVVo.addWidget(self.Buttons)
        VVVs.addWidget(self.Buttont)
        VVVs.addWidget(self.Buttonf)
        
    ###均值滤波器
    def ls(self):
        ls_img =cv2.blur(self.img, (7, 7))

        cv2.namedWindow('ls',cv2.WINDOW_NORMAL)
        cv2.imshow('ls',ls_img)
        self.close()
    
    ###线性锐化滤波器
    def kernel(self):
        kernel_sharpen_1 = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]])
        
        lr_img = cv2.filter2D(self.img, -1, kernel_sharpen_1)
        cv2.namedWindow('lr',cv2.WINDOW_NORMAL)
        cv2.imshow('lr',lr_img)
        self.close()

    ###非线性平滑滤波器
    def nls(self):
        nls_img = cv2.medianBlur(self.img, 5)
        cv2.namedWindow('nls',cv2.WINDOW_NORMAL)
        cv2.imshow('nls',nls_img)
        self.close()

    ###非线性锐化滤波器
    def nlr(self):
        nlr_img = cv2.bilateralFilter(self.img, 5, 31, 31)
        cv2.namedWindow('nlr',cv2.WINDOW_NORMAL)
        cv2.imshow('nlr',nlr_img)
        self.close()


class Kernel(QWidget):
    def __init__(self,image):
        super().__init__()
        self.img = image
        self.initUI()

    def initUI(self):
        Hlayout = QHBoxLayout(self)
        Vlayout = QVBoxLayout(self)

        self.labelo = QLabel('请输入卷积核大小,x轴方向大小',self)
        self.labels = QLabel('请输入卷积核大小,y轴方向大小',self)

        self.lineo = QLineEdit(self)
        self.lines = QLineEdit(self)

        self.pusho = QPushButton('确认',self)
        self.pushs = QPushButton('取消',self,clicked=self.close)

        Hlayout.addWidget(self.labelo)
        Hlayout.addWidget(self.lineo)
        Hlayout.addWidget(self.labels)
        Hlayout.addWidget(self.lines)
        Hlayout.addLayout(Vlayout)
        Vlayout.addWidget(self.pusho)
        Vlayout.addWidget(self.pushs)

class KKBinarry(Kernel):
    def __init__(self, image):
        super().__init__(image)

        self.pusho.clicked.connect(self.Binarryy)

    def Binarryy(self):
        MoveX = int(self.lineo.text())
        MoveY = int(self.lines.text())
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MoveX, MoveY))

        self.binar = Binaryy(self.img,self.kernel)
        self.binar.show()
        self.close()

class KKBinaryApplicee(Kernel):
    def __init__(self, image):
        super().__init__(image)
        self.pusho.clicked.connect(self.BinaryApplicee)

    def BinaryApplicee(self):
        MoveX = int(self.lineo.text())
        MoveY = int(self.lines.text())
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MoveX, MoveY))

        self.biapp = BinaryApplice(self.img,self.kernel)
        self.biapp.show()
        self.close()

class KKGrayXing(Kernel):
    def __init__(self, image):
        super().__init__(image)
        self.pusho.clicked.connect(self.GrayXing)

    def GrayXing(self):
        MoveX = int(self.lineo.text())
        MoveY = int(self.lines.text())
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MoveX, MoveY))

        self.eee  = GrayXingg(self.img,self.kernel)
        self.eee.show()

class KKGrayApp(Kernel):
    def __init__(self, image):
        super().__init__(image)
        self.img = image
        self.pusho.clicked.connect(self.GrayAppp)

    def GrayAppp(self):
        MoveX = int(self.lineo.text())
        MoveY = int(self.lines.text())
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MoveX, MoveY))
        self.www = GrayApp(self.img,self.kernel)
        self.www.show()


class GrayApp(QWidget):
    def __init__(self,image,kernel):
        super().__init__()
        self.img = image
        self.kernel = kernel
        self.initUI()

    def initUI(self):
        Vlayout = QVBoxLayout(self)
        Hlayout = QHBoxLayout(self)
        HHlayout = QHBoxLayout(self)

        self.pusho = QPushButton('形态梯度',self,clicked=self.GDD)
        self.pushs = QPushButton('形态平滑',self,clicked=self.SMM)
        self.pusht = QPushButton('高帽变换',self,clicked=self.HATG)
        self.pushf = QPushButton('低帽变换',self,clicked=self.HATB)

        Vlayout.addLayout(Hlayout)
        Vlayout.addLayout(HHlayout)

        Hlayout.addWidget(self.pusho)
        Hlayout.addWidget(self.pushs)
        HHlayout.addWidget(self.pusht)
        HHlayout.addWidget(self.pushf)

    def GDD(self):
        Gd_out = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, self.kernel)
        cv2.namedWindow('GDD',cv2.WINDOW_NORMAL)
        cv2.imshow('GDD',Gd_out)

    def SMM(self):
        Smm = cv2.boxFilter(self.img,-1,(3,3),normalize=True)
        cv2.namedWindow('SMM',cv2.WINDOW_NORMAL)
        cv2.imshow('SMM',Smm)

    def HATG(self):
        hatg = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, self.kernel)
        cv2.namedWindow('HATG',cv2.WINDOW_NORMAL)
        cv2.imshow('HATG',hatg)

    def HATB(self):
        ret, binary = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        hatb = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, self.kernel)
        cv2.namedWindow('HATB',cv2.WINDOW_NORMAL)
        cv2.imshow('HATB',hatb)




class Binaryy(QWidget):
    def __init__(self,image,kernel):
        super().__init__()
        self.img = image
        self.kernel = kernel
        self.initUI()

    def initUI(self):
        Hlayout = QHBoxLayout(self)
        Vlayout = QVBoxLayout(self)
        VVlayout = QVBoxLayout(self)

        self.pusho = QPushButton('膨胀',self,clicked=self.dilated)
        self.pushs = QPushButton('腐蚀',self,clicked=self.blur)
        self.pusht = QPushButton('开操作',self,clicked=self.opening)
        self.pushf = QPushButton('闭操作',self,clicked=self.closing)

        Hlayout.addLayout(Vlayout)
        Hlayout.addLayout(VVlayout)

        Vlayout.addWidget(self.pusho)
        Vlayout.addWidget(self.pushs)
        VVlayout.addWidget(self.pusht)
        VVlayout.addWidget(self.pushf)



    def dilated(self):
        dilated = cv2.dilate(self.img,self.kernel)
        cv2.namedWindow('Binary',cv2.WINDOW_NORMAL)
        cv2.imshow('Binary',dilated)

    def blur(self):
        blurr = cv2.blur(self.img,(5,5))
        cv2.namedWindow('blurr',cv2.WINDOW_NORMAL)
        cv2.imshow('blurr',blurr)

    def opening(self):
        opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel)
        cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
        cv2.imshow('opening',opening)

    def closing(self):
        closing = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, self.kernel)
        cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
        cv2.imshow('closing',closing)






class Teshuu(Kernel):
    def __init__(self, image):
        super().__init__(image)
        self.img = image
        self.pusho.clicked.connect(self.confirm)

    ### 矩阵乘法
    def decreaseArray(self,image1, image2):
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
    def increaseArray(self,image1, image2):
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
    

    def confirm(self):
        MoveX = int(self.lineo.text())
        MoveY = int(self.lines.text())
        imageAver3 = cv2.blur(self.img,(MoveX,MoveY))

        upsharpMask = self.decreaseArray(self.img,imageAver3)
        imageSharp = self.increaseArray(self.img,upsharpMask)
            
        himage = np.hstack([self.img,imageAver3])
        hhimage = np.hstack([imageSharp,upsharpMask])
        vimage = np.vstack([himage,hhimage])

        cv2.namedWindow('Teshu',cv2.WINDOW_NORMAL)
        cv2.imshow('Teshu',vimage)
        




class BinaryApplice(QWidget):
    def __init__(self,image,kernel):
        super().__init__()
        self.img = image
        self.kernel = kernel
        self.initUI()

    def initUI(self):
        Hlayout = QHBoxLayout(self)
        
        self.pusho = QPushButton('去噪',self,clicked=self.noise)
        self.pushs = QPushButton('目标检测',self,clicked=self.dection)

        Hlayout.addWidget(self.pusho)
        Hlayout.addWidget(self.pushs)

    def noise(self):
        noise = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel)
        cv2.namedWindow('noise',cv2.WINDOW_NORMAL)
        cv2.imshow('noise',noise)

    def dection(self):
        gradient = cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, self.kernel)
        cv2.namedWindow('dection',cv2.WINDOW_NORMAL)
        cv2.imshow('dection',gradient)

class GrayXingg(QWidget):
    def __init__(self,image,kernel):
        super().__init__()
        self.img = image
        self.kernel = kernel
        self.initUI()

    def initUI(self):

        Hlayout = QHBoxLayout(self)
        Vlayout = QVBoxLayout(self)
        VVlayout = QVBoxLayout(self)

        self.pusho = QPushButton('腐蚀',self,clicked=self.eroded)
        self.pushs = QPushButton('膨胀',self,clicked=self.dilation)
        self.pusht = QPushButton('开运算',self,clicked=self.opening)
        self.pushf = QPushButton('闭运算',self,clicked=self.closing)

        Hlayout.addLayout(Vlayout)
        Hlayout.addLayout(VVlayout)

        Vlayout.addWidget(self.pusho)
        Vlayout.addWidget(self.pushs)

        VVlayout.addWidget(self.pusht)
        VVlayout.addWidget(self.pushf)

    def eroded(self):
        eroded = cv2.erode(self.img,self.kernel,iterations = 1)
        cv2.namedWindow('erode',cv2.WINDOW_NORMAL)
        cv2.imshow('erode',eroded)

    def dilation(self):
        dilationed = cv2.dilate(self.img,self.kernel,iterations = 1)
        cv2.namedWindow('dilation',cv2.WINDOW_NORMAL)
        cv2.imshow('dilation',dilationed)

    def opening(self):
        opening = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel)
        cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
        cv2.imshow('opening',opening)

    def closing(self):
        closing = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, self.kernel)
        cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
        cv2.imshow('closing',closing)

class KKHighLow(QWidget):
    def __init__(self,image):
        super().__init__()
        self.img = image

        Vlayout = QVBoxLayout(self)
        Hlayout = QHBoxLayout(self)

        self.pusho = QPushButton('低通',self,clicked= self.High)
        self.pushs = QPushButton('高通',self,clicked = self.Low)
        Vlayout.addLayout(Hlayout)
        Hlayout.addWidget(self.pusho)
        Hlayout.addWidget(self.pushs)
        self.LH = ''

    def High(self):
        self.LH = 'lp'
        self.eee = HighLow(self.img,self.LH)
        self.eee.show()
        self.close()

    def Low(self):
        self.LH = 'hp'
        self.eee = HighLow(self.img,self.LH)
        self.eee.show()
        self.close()



class HighLow(QWidget):
    def __init__(self,image,LH):
        super().__init__()
        self.img = image
        self.LH = LH

        self.D0=30
        self.W=0
        self.N=2

        ### 离散傅里叶变换
        dft = cv2.dft(np.float32(self.img),flags=cv2.DFT_COMPLEX_OUTPUT)
        ### 中心化
        self.dtf_shift=np.fft.fftshift(dft) 
    
        self.rows,self.cols = self.img.shape[0],self.img.shape[1]
        self.crow,self.ccol = int(self.rows / 2),int(self.cols/2) ### 计算频谱中心




        Vlayour = QVBoxLayout(self)
        Hlayour = QHBoxLayout(self)
        HHlayour = QHBoxLayout(self)

        self.pusho = QPushButton('巴特沃滤波器',self,clicked=self.BT)
        self.pushs = QPushButton('理想滤波器',self,clicked=self.ideal)
        self.pusht = QPushButton('指数滤波器',self,clicked=self.expose)

        Vlayour.addLayout(Hlayour)
        Vlayour.addLayout(HHlayour)
        Hlayour.addWidget(self.pusho)
        Hlayour.addWidget(self.pushs)
        HHlayour.addWidget(self.pusht)



    def BT(self):
        mask=np.ones((self.rows,self.cols,2)) #生成rows行cols列的2维矩阵
        for i in range(self.rows):
            for j in range(self.cols):
                D = np.sqrt((i-self.crow)**2+(j-self.ccol)**2)
                if(self.LH == 'lp'):
                    mask[i, j] = 1/(1+(D/self.D0)**(2*self.N))
                elif(self.LH == 'hp'):
                    mask[i, j] = 1/(1+(self.D0/D)**(2*self.N))
                else:
                    assert('type error')

        fshift = self.dtf_shift*mask

        f_ishift=np.fft.ifftshift(fshift) 
        bimg_back=cv2.idft(f_ishift) 
        bimg_back=cv2.magnitude(bimg_back[:,:,0],bimg_back[:,:,1]) #计算像素梯度的绝对值
        bimg_back=np.abs(bimg_back)
        bimg_back=(bimg_back-np.amin(bimg_back))/(np.amax(bimg_back)-np.amin(bimg_back))
        cv2.namedWindow('BT',cv2.WINDOW_NORMAL)
        cv2.imshow('BT',bimg_back)



    def ideal(self):
        mask=np.ones((self.rows,self.cols,2)) #生成rows行cols列的2维矩阵
        for i in range(self.rows):
            for j in range(self.cols):
                D = np.sqrt((i-self.crow)**2+(j-self.ccol)**2)
                if(self.LH == 'lp'):
                    if(D > self.D0):
                        mask[i, j] = 0
                elif(self.LH == 'hp'):
                    if(D <self.D0):
                        mask[i, j] = 0
                else:
                    assert('type error')

        fshift = self.dtf_shift*mask

        if_ishift=np.fft.ifftshift(fshift) 
        iimg_back=cv2.idft(if_ishift) 
        iimg_back=cv2.magnitude(iimg_back[:,:,0],iimg_back[:,:,1]) #计算像素梯度的绝对值
        iimg_back=np.abs(iimg_back)
        iimg_back=(iimg_back-np.amin(iimg_back))/(np.amax(iimg_back)-np.amin(iimg_back))
        cv2.namedWindow('ideal',cv2.WINDOW_NORMAL)
        cv2.imshow('ideal',iimg_back)
        

    def expose(self):
        mask=np.ones((self.rows,self.cols,2)) #生成rows行cols列的2维矩阵
        for i in range(self.rows):
            for j in range(self.cols):
                D = np.sqrt((i-self.crow)**2+(j-self.ccol)**2)
                if(self.LH == 'lp'):
                    mask[i, j] = np.exp(-(D/self.D0)**(2*self.N))
                elif(self.LH == 'hp'):
                    mask[i, j] = np.exp(-(self.D0/D)**(2*self.N))
                else:
                    assert('type error')

        efshift = self.dtf_shift*mask

        ef_ishift=np.fft.ifftshift(efshift) 
        eimg_back=cv2.idft(ef_ishift) 
        eimg_back=cv2.magnitude(eimg_back[:,:,0],eimg_back[:,:,1]) #计算像素梯度的绝对值
        eimg_back=np.abs(eimg_back)
        eimg_back=(eimg_back-np.amin(eimg_back))/(np.amax(eimg_back)-np.amin(eimg_back))
        cv2.namedWindow('expose',cv2.WINDOW_NORMAL)
        cv2.imshow('expose',eimg_back)

class Fourierrr():
    def __init__(self,image):
        super().__init__()
        self.img = image

    def initi(self):
        ### 二维傅里叶变换，进行频率转换
        fo = np.fft.fft2(self.img)
        
        ### 零频率分量(DC分量)位于左上角，如果要使其居中，则使用fftshift将结果进行偏移
        fshift = np.fft.fftshift(fo)
        fo_img = np.log(np.abs(fshift))
        
        ### 傅里叶逆变换,iffshift反向位移，使(DC分量)再次出现在左上角
        f1shift = np.fft.ifftshift(fshift)
        nfo_img = np.fft.ifft2(f1shift)
        nfo_img = np.abs(nfo_img)

        plt.figure()
        plt.subplot(121), plt.imshow(fo_img, cmap='gray')
        plt.title('fo_img'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(nfo_img, cmap='gray')
        plt.title('Ifo_img'), plt.xticks([]), plt.yticks([])
        plt.show()

class Zuhelubooo():
    def __init__(self,image):
        super().__init__()
        self.img = image

    def initi(self):
            f = np.fft.fft2(self.img)
            fshift = np.fft.fftshift(f)
            s1 = np.log(np.abs(fshift))

            w , h = self.img.shape
            flt = np.zeros(self.img.shape)

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

            plt.figure()
            plt.subplot(221), plt.imshow(self.img, cmap='gray')
            plt.xticks([]), plt.yticks([])

            plt.subplot(222), plt.imshow(s1, cmap='gray')
            plt.xticks([]), plt.yticks([])

            plt.subplot(223), plt.imshow(flt, cmap='gray')
            plt.xticks([]), plt.yticks([])

            plt.subplot(224), plt.imshow(new_img, cmap='gray')
            plt.xticks([]), plt.yticks([])

            plt.show()


class CWuYueSuuu():
    def __init__(self,image):
        super().__init__()
        self.img = image

    def initi(self):
        ### 傅里叶变换
        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
        ### 使用shift函数将频率为0的部分，从左上角移动到中心位置
        dftshift = np.fft.fftshift(dft)
        ### 由于输出的频谱结果是一个复数，需要调用cv.magnitude()函数将傅里叶变换的双通达结果转换为0到255的范围。
        res1= 20*np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))

        ### 傅里叶逆变换
        ishift = np.fft.ifftshift(dftshift)
        iimg = cv2.idft(ishift)
        res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

        plt.figure()
        plt.subplot(121), plt.imshow(res1, cmap='gray')
        plt.title('Dft-Process'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(res2, cmap='gray')
        plt.title('Ishift-Process'), plt.xticks([]), plt.yticks([])
        plt.show()






class KKBSP(QWidget):
    def __init__(self,image):
        super().__init__()
        self.img = image

        Vlayout = QVBoxLayout(self)
        Hlayout = QHBoxLayout(self)

        self.pusho = QPushButton('带通',self,clicked= self.BPH)
        self.pushs = QPushButton('带阻',self,clicked = self.SPH)
        Vlayout.addLayout(Hlayout)
        Hlayout.addWidget(self.pusho)
        Hlayout.addWidget(self.pushs)
        self.BS = ''

    def BPH(self):
        self.BS = 'BP'
        self.eee = BSP(self.img,self.BS)
        self.eee.show()
        self.close()

    def SPH(self):
        self.BS = 'SP'
        self.eee = BSP(self.img,self.BS)
        self.eee.show()
        self.close()



class BSP(QWidget):
    def __init__(self,image,BS):
        super().__init__()
        self.img = image
        self.BS = BS

        self.D0=30
        self.W=0
        self.N=2

        ### 离散傅里叶变换
        dft = cv2.dft(np.float32(self.img),flags=cv2.DFT_COMPLEX_OUTPUT)
        ### 中心化
        self.dtf_shift=np.fft.fftshift(dft) 
    
        self.rows,self.cols = self.img.shape[0],self.img.shape[1]
        self.crow,self.ccol = int(self.rows / 2),int(self.cols/2) ### 计算频谱中心
 
        Vlayour = QVBoxLayout(self)
        Hlayour = QHBoxLayout(self)
        HHlayour = QHBoxLayout(self)

        self.pusho = QPushButton('巴特沃滤波器',self,clicked=self.BT)
        self.pushs = QPushButton('理想滤波器',self,clicked=self.ideal)
        self.pusht = QPushButton('指数滤波器',self,clicked=self.expose)

        Vlayour.addLayout(Hlayour)
        Vlayour.addLayout(HHlayour)
        Hlayour.addWidget(self.pusho)
        Hlayour.addWidget(self.pushs)
        HHlayour.addWidget(self.pusht)

    def BT(self):
        mask=np.ones((self.rows,self.cols,2)) #生成rows行cols列的2维矩阵
        for i in range(self.rows):
            for j in range(self.cols):
                D = np.sqrt((i-self.crow)**2+(j-self.ccol)**2)
                if(self.BS == 'BP'):
                    mask[i, j] = 1/(1+(D*self.W/(D**2-self.D0**2+1))**(2*self.N))
                elif(self.BS == 'SP'):
                    mask[i, j] = 1/(1+((D**2-self.D0**2)/D*self.W+1)**(2*self.N))
                else:
                    assert('type error')

        fshift = self.dtf_shift*mask

        f_ishift=np.fft.ifftshift(fshift) 
        bimg_back=cv2.idft(f_ishift) 
        bimg_back=cv2.magnitude(bimg_back[:,:,0],bimg_back[:,:,1]) #计算像素梯度的绝对值
        bimg_back=np.abs(bimg_back)
        bimg_back=(bimg_back-np.amin(bimg_back))/(np.amax(bimg_back)-np.amin(bimg_back))
        cv2.namedWindow('BT',cv2.WINDOW_NORMAL)
        cv2.imshow('BT',bimg_back)



    def ideal(self):
        mask=np.ones((self.rows,self.cols,2)) #生成rows行cols列的2维矩阵
        for i in range(self.rows):
            for j in range(self.cols):
                D = np.sqrt((i-self.crow)**2+(j-self.ccol)**2)
                if(self.BS == 'BP'):
                    if(D > self.D0 and D < self.D0+self.W):
                        mask[i, j] = 0
                elif(self.BS == 'SP'):
                    if(D < self.D0 and D > self.D0+self.W):
                        mask[i, j] = 0
                else:
                    assert('type error')

        fshift = self.dtf_shift*mask

        if_ishift=np.fft.ifftshift(fshift) 
        iimg_back=cv2.idft(if_ishift) 
        iimg_back=cv2.magnitude(iimg_back[:,:,0],iimg_back[:,:,1]) #计算像素梯度的绝对值
        iimg_back=np.abs(iimg_back)
        iimg_back=(iimg_back-np.amin(iimg_back))/(np.amax(iimg_back)-np.amin(iimg_back))
        cv2.namedWindow('ideal',cv2.WINDOW_NORMAL)
        cv2.imshow('ideal',iimg_back)
        

    def expose(self):
        mask=np.ones((self.rows,self.cols,2)) #生成rows行cols列的2维矩阵
        for i in range(self.rows):
            for j in range(self.cols):
                D = np.sqrt((i-self.crow)**2+(j-self.ccol)**2)
                if(self.BS == 'BP'):
                    mask[i, j] = np.exp(-(D*self.W/(D**2 - self.D0**2+1))**(2*self.N))
                elif(self.BS == 'SP'):
                    mask[i, j] = np.exp(-((D**2 - self.D0**2)/D*self.W+1)**(2*self.N))
                else:
                    assert('type error')

        efshift = self.dtf_shift*mask

        ef_ishift=np.fft.ifftshift(efshift) 
        eimg_back=cv2.idft(ef_ishift) 
        eimg_back=cv2.magnitude(eimg_back[:,:,0],eimg_back[:,:,1]) #计算像素梯度的绝对值
        eimg_back=np.abs(eimg_back)
        eimg_back=(eimg_back-np.amin(eimg_back))/(np.amax(eimg_back)-np.amin(eimg_back))
        cv2.namedWindow('expose',cv2.WINDOW_NORMAL)
        cv2.imshow('expose',eimg_back)