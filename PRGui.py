import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Gui(object):

    # def __init__(self):
    #     super().__init__()
    #     self.initUI()


    def initUI(self,window): 
        window.resize(1000,600)

        ### 设置窗口标题
        self.setWindowTitle('图像处理工程作业')  

        ### 将centralWidget设置为依附于self的一个子部件        
        self.centralWidget = QtWidgets.QWidget(window)

        ### 全局布局,在中心部件中设置布局
        mainLayout = QVBoxLayout(self.centralWidget)

        ### 顶部布局
        topLayout = QHBoxLayout()
        self.label = QLabel()
        self.show()

        ### self设置中心部件
        self.setCentralWidget(self.centralWidget)
        topLayout.setSpacing(5)
        #################################################################
        self.FileUpButton = QPushButton('文件',self)
        topLayout.addWidget(self.FileUpButton)

        Fileup = QMenu(self)
        self.openAct = QAction('打开',self)
        # self.saveAct = QAction('保存',self)
        self.exitAct = QAction('退出',self)
        Fileup.addActions((self.openAct,self.exitAct))
        self.FileUpButton.setMenu(Fileup)

        #################################################################
        self.XYButton = QPushButton('坐标变换',self)
        topLayout.addWidget(self.XYButton)

        XYMenu = QMenu(self)
        self.MoveAct = QAction('平移变换',self) 
        self.ScaleAct = QAction('尺度变换',self)
        self.RotateAct = QAction('旋转变换',self)
        self.AffineAct = QAction('仿射变换',self)

        XYMenu.addActions((self.MoveAct,self.ScaleAct,self.RotateAct,self.AffineAct))
        self.XYButton.setMenu(XYMenu)
    
        #################################################################
        self.SpatialButton = QPushButton('空域变换处理',self)
        topLayout.addWidget(self.SpatialButton)

        Spatial = QMenu(self)

        self.GrayMenu = QAction('灰度值映射',self)
        # self.BackActAct = QAction('求反', self) 
        # self.CompressAct = QAction('动态范围压缩',self)
        # self.StairsAct = QAction('阶梯量化',self)
        # self.ThreshAct = QAction('阈值分割',self)
        # GrayMenu.addActions((self.BackActAct,self.CompressAct,self.StairsAct,self.ThreshAct))
        Spatial.addAction(self.GrayMenu)
        

        #####
        self.Math = QAction('算术运算',self)
        # self.AddAct = QAction('加法',self)
        # self.Mean = QAction('平均法消除噪声',self)
        # self.Stract = QAction('减法',self)
        # Math.addActions((self.AddAct,self.Mean,self.Stract))
        Spatial.addAction(self.Math)

        #####
        self.Histogram = QAction('直方图修正',self)
        # self.MeanAct = QAction('直方图均衡化',self)
        # self.RuglarAct = QAction('直方图规定化',self)
        # Histogram.addAction(self.MeanAct)
        # Histogram.addAction(self.RuglarAct)
        Spatial.addAction(self.Histogram)

        #####
        self.FilterAct = QAction('空域滤波',self)
        # self.Slide = QAction('线性平滑滤波器',self)
        # self.LineRui = QAction('线性锐化滤波器',self)
        # self.NLineSlide = QAction('非线性平滑滤波器',self)
        # self.NLineRui = QAction('非线性锐化滤波器',self)
        # FilterAct.addActions((self.Slide,self.LineRui,self.NLineSlide,self.NLineRui))
        Spatial.addAction(self.FilterAct)

        self.SpatialButton.setMenu(Spatial)


        ##############################################################
        self.FrequencyButton = QPushButton('频域图像增强',self)
        topLayout.addWidget(self.FrequencyButton)

        Frequency = QMenu(self)
        self.Founrier = QAction('傅里叶变换和反变换',self)

        self.ideal = QAction('理想滤波器',self)
        self.butterworth = QAction('butterworth滤波器',self)
        self.exponential = QAction('指数滤波器',self)

        self.TeShu = QAction('特殊高通滤波器',self)
        self.DaiTZTong = QAction('带通带阻滤波器',self)
        self.TongTai = QAction('同态滤波器',self)
        Frequency.addActions((self.Founrier,self.ideal,self.butterworth,self.exponential,self.TeShu,self.DaiTZTong,self.TongTai))

        self.FrequencyButton.setMenu(Frequency)
        
        ##############################################################
        self.CuringButton = QPushButton('图像恢复',self)
        topLayout.addWidget(self.CuringButton)

        Curing = QMenu(self)
        #########
        self.KongYu = QAction('空域噪声滤波器',self)
        # self.Meanlubo = QAction('均值滤波器',self)
        # self.Sequencelubo = QAction('动态滤波器',self)
        # KongYu.addAction(self.Meanlubo)
        # KongYu.addAction(self.Sequencelubo)
        #########
        self.Zuhelubo = QAction('均值滤波器',self)
        # self.HunHelubo = QAction('混合滤波器',self)
        # self.XuanZelubo = QAction('选择滤波器',self)
        # Zuhelubo.addAction(self.HunHelubo)
        # Zuhelubo.addAction(self.XuanZelubo)
        #########
        self.WuYuanSu = QAction('无约束滤波器',self)
        # self.Nilubo = QAction('逆滤波',self)
        # WuYuanSu.addAction(self.Nilubo)
        #########
        self.YouYueSu = QAction('有约束滤波器',self)
        # self.NaWei = QAction('纳维滤波器',self)
        # YouYueSu.addAction(self.NaWei)

        Curing.addAction(self.KongYu)
        Curing.addAction(self.Zuhelubo)
        Curing.addAction(self.WuYuanSu)
        Curing.addAction(self.YouYueSu)

        self.CuringButton.setMenu(Curing)
        

        ######################################################
        self.PitCodeButton = QPushButton('图像编码',self)
        topLayout.addWidget(self.PitCodeButton)

        PitCode = QMenu(self)
        ChangeCode = QMenu('变长编码',self)
        self.HaFucode = QAction('哈夫曼编码',self)
        self.GeLunBucode = QAction('哥伦布编码',self)
        self.XiangNong = QAction('香农-法诺编码',self)
        self.MathCode = QAction('算数编码',self)

        self.WeiPingMian = QAction('位平面编码',self)
        ChangeCode.addActions((self.HaFucode,self.GeLunBucode,self.XiangNong,self.MathCode))
        PitCode.addMenu(ChangeCode)
        PitCode.addAction(self.WeiPingMian)

        self.PitCodeButton.setMenu(PitCode)
        

        ######################################################
        self.CodeStandButton = QPushButton('图像编码技术和标准',self)
        topLayout.addWidget(self.CodeStandButton)

        CodeStand = QMenu(self)
        self.DPCMCode = QAction('DPCM编码',self)
        self.Cose = QAction('余弦变换编码',self)
        self.XiaoWave = QAction('小波变换编码',self)
        CodeStand.addActions((self.DPCMCode,self.Cose,self.XiaoWave))
        
        self.CodeStandButton.setMenu(CodeStand)
 
        ####################################################
        self.PicCutButton = QPushButton('图像分割',self)
        topLayout.addWidget(self.PicCutButton)

        PicCut = QMenu(self)
        self.DyaCut = QAction('动态分割',self)
        self.Singal = QAction('单阈值分割',self)
        PicCut.addActions((self.DyaCut,self.Singal))
        self.PicCutButton.setMenu(PicCut)

        #######################################################
        self.TpyeCutButton = QPushButton('典型分割',self)
        TpyeCut = QMenu(self)
        topLayout.addWidget(self.TpyeCutButton)

        self.SUSAN = QAction('SUSAN边缘检测',self)
        self.First = QAction('主动轮廓',self)
        self.FenWaterCut = QAction('分水岭分割',self)
        TpyeCut.addActions((self.SUSAN,self.First,self.FenWaterCut))

        self.TpyeCutButton.setMenu(TpyeCut)

        ##########################################################
        self.MathMove = QPushButton('数学形态学',self)
        topLayout.addWidget(self.MathMove)

        MathMove = QMenu(self)
        
        ######
        self.Binary = QAction('二值形态学',self)
        # self.FuShi = QAction('腐蚀',self)
        # self.PengZhang = QAction('膨胀',self)
        # self.KaiQi = QAction('开启',self)
        # self.BiHe = QAction('闭合',self)
        # Binary.addActions((self.FuShi,self.PengZhang,self.KaiQi,self.BiHe))

        ######
        self.BinaryApplice = QAction('二值形态学应用',self)
        # self.ZaoSheng = QAction('噪声消除',self)
        # self.MuBiao = QAction('目标检测',self)
        # self.QuYu = QAction('区域填充',self)
        # BinaryApplice.addActions((self.ZaoSheng,self.MuBiao,self.QuYu))

        #######
        self.GrayXing = QAction('灰度形态学',self)
        # self.GFuShi = QAction('腐蚀',self)
        # self.GPengZhang = QAction('膨胀',self)
        # self.GKaiQi = QAction('开启',self)
        # self.GBiHe = QAction('闭合',self)
        # GrayXing.addActions((self.GFuShi,self.GPengZhang,self.GKaiQi,self.GBiHe))

        ########
        self.GrayXingAp = QAction('基于灰度形态学的应用',self)
        # self.XingTiDu = QAction('形态梯度',self)
        # self.XingPingH = QAction('高帽变换',self)
        # self.XingDiH = QAction('低帽变换',self) 
        # GrayXingAp.addActions((self.XingTiDu,self.XingPingH,self.XingDiH))


        MathMove.addAction(self.Binary)
        MathMove.addAction(self.BinaryApplice)
        MathMove.addAction(self.GrayXing)
        MathMove.addAction(self.GrayXingAp)

        self.MathMove.setMenu(MathMove)
        #########################################################
        self.MathsonButton = QPushButton('算子',self)
        topLayout.addWidget(self.MathsonButton)      

        Mathson = QMenu(self)
        self.Sobel = QAction('Sobel算子',self)
        self.Roberts = QAction('Roberts算子',self)
        self.LPLS = QAction('拉普拉斯算子',self)
        self.Canny = QAction('Canny算子',self)
        self.Prewitt = QAction('Prewitt算子',self)
        self.HLPLS = QAction('高拉普拉斯算子',self)
        Mathson.addActions((self.Sobel,self.Roberts,self.LPLS,self.Canny,self.Prewitt,self.HLPLS))

        self.MathsonButton.setMenu(Mathson)
        #############################################################
        #####中间布局
        midLayout = QHBoxLayout()
        self.showImageView = QTableWidget()
        midLayout.addWidget(self.showImageView)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(midLayout)

        ####设置stretch
        mainLayout.setStretchFactor(topLayout,1)
        mainLayout.setStretchFactor(midLayout,6)

        ############################################################




# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = Gui()
#     sys.exit(app.exec_())