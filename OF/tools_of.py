#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:21:36 2022

@author: xuyang
"""

from skimage.transform import warp, SimilarityTransform,rotate
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob

def fitswrite(fileout, im, header=None):
    from astropy.io import fits
    import os
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:        
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    from astropy.io import fits
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head 
    
def imrotate(im,para):
    return rotate(im,para,mode='reflect')
    
def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2

def num_time(obstime):
    #GST Obs_time to number
    Th = obstime[0:2]
    Tm = obstime[3:5]
    if obstime[7] == '.':
        Ts=obstime[6]
    else:
        Ts=obstime[6:8]
    NT = 3600*int(Th)+60*int(Tm)+int(Ts)
    return NT

def zscore2(im):
    im = (im - np.mean(im)) / im.std()
    return im  

def toMP4(mp4name,jpgdir='JPG'):
    filelist=sorted(glob.glob(jpgdir+'*.jpg'))
    frame = cv2.imread(filelist[0])

    fps = 10.0 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    size = (frame.shape[1]-10,frame.shape[0]-10) 
    out = cv2.VideoWriter(mp4name+'.mp4', fourcc, fps, size) 

    for image_file in filelist:
        frame = cv2.imread(image_file)
        out.write(frame[:size[1],:size[0],:])
        print(image_file)
     
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break    
     
    out.release()
    cv2.destroyAllWindows()
    
def array2movie(imgarr,movie_name,tmp='/home/xuyang/tmp/',title_cube=0,color='gray'):
    import os
    from matplotlib import pyplot as plt
    import shutil
    if not os.path.exists(tmp):
        os.makedirs(tmp) 
    ni,h,w=imgarr.shape
    K=0
    for i in range(ni):
        im=imgarr[i,:,:]
#    IM=tools.tocolor2(im)
        if title_cube == 0:
            subfile="{:0>3d}".format(i)
        else:
            subfile=title_cube[i]
        
        mtmp=zscore2(im)
        if K==0:            
            dis=plt.imshow(mtmp,vmax=4,vmin=-4,cmap=color,origin='lower')      
            plt.pause(0.1)
            plt.draw()
        else:
            dis.set_data(mtmp)
            plt.pause(0.1)
            plt.draw()
        K=1
        plt.title(subfile)
        print(i,subfile)
        plt.savefig(tmp+subfile+'.jpg',dpi=300)
       
    mp4n=movie_name
    toMP4(mp4n,jpgdir=tmp) 
    shutil.rmtree(tmp)
    
def immove2(im,dx=0,dy=0):
    im2=im.copy()
    tform = SimilarityTransform(translation=(dx,dy))
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='constant',cval=0)
    return im2

def cubealign(im,wd=100,winsize=21,step=2):
    avg=im[:,wd:-wd,wd:-wd].mean()
    im=im/im[:,wd:-wd,wd:-wd].mean(axis=(-2,-1))[:,np.newaxis,np.newaxis]*avg
    IM=np.mean(im,axis=0)
    for j in range(step):
        im2=[]
        for i in range(im.shape[0]):
            d,model,flag,flow =align_opflow(IM[wd:-wd,wd:-wd],im[i,wd:-wd,wd:-wd],winsize=winsize,step=5,r_t=5)
            print('shift:',j,i,d,round(flag,3))#,model_robust.scale)
           
            tmp=immove2(im[i],-d[0],-d[1])
            im2.append(tmp)
        im2=np.array(im2)
        im2=im2.astype(np.float32)
        IM=np.mean(im2,axis=0)     
    return(im2)  
    
def align_opflow(im1org,im2org,winsize=11,step=5,r_t=5,arrow=0):
    wd=500
    
    im1=im1org[wd:-wd,wd:-wd].copy()
    im2=im2org[wd:-wd,wd:-wd].copy()


    #计算光流#Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(im1, im2, flow=None, pyr_scale=0.5, levels=5, winsize=winsize, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)
    
    #按照step 选取有效参考点#Select effective points based on step setup
    h,w=im1.shape
    x1, y1 = np.meshgrid(np.arange(w), np.arange(h))
    x2=x1.astype('float32')+flow[:,:,0]
    y2=y1.astype('float32')+flow[:,:,1]
    x1=x1[::step,::step]
    y1=y1[::step,::step]
    x2=x2[::step,::step]
    y2=y2[::step,::step]

    
    src=(np.vstack((x1.flatten(),y1.flatten())).T) #源坐标#source coordinates
    dst=(np.vstack((x2.flatten(),y2.flatten())).T) #目标坐标#destination coordinates
    s=dst-src #位移量#displacement
    Dlt0=((np.abs(s[:,0])>0) + (np.abs(s[:,1])>0))>0 #判断是否无位移#Any displacement here?

    if Dlt0.sum()>0: #处理有位移的图像#for displaced images
        dst=dst[Dlt0]
        src=src[Dlt0]
        s=s[Dlt0]
        #####筛选有效参考点################Select effective reference points
        d, D= mode_model((src,dst),  residual_threshold=r_t) #统计系统性偏移量#calculate systematic displacements
        ###########如果需要，画光流场##################Draw OF map if needed
        if arrow==1:
            plt.figure()
            showim(im1)
            x=src[D,0]
            y=src[D,1]
            fx=s[D,0]
            fy=-s[D,1]
            plt.quiver(x,y,fx,fy,color='r',scale=0.1,scale_units='dots',minshaft=2)
        ###########Draw OF map if needed##################
        #####Outputs，
        #1，Ratio of effective area in the FOV；
        #2，FLAG，population of effective points; <=1, better >0.1; used as weight for cross alignment between wavelengths 
        #3，Number of effctive points, expect to be more than 1000, less than 200 may be bad; uniformity is also important
        #4，calculated displacement
        #5，accuracy of calculated displacement
        flag=D.sum()/Dlt0.sum() #有效控制点占比， 用于评价配准的概率#population of the effective points
        print(round(flag,3),D.sum(),d,np.std(s[D],axis=0)/np.sqrt(D.sum()-1))#,model_robust.scale)
        
        model=1 #是否有位移的图像 # mark for displaced image

    else:
        d=[0,0]
        model=0 #完全相同的图像 #mark for identical image
        flag=0

    
    return d,model,flag,flow 

def showim(im,k=3,cmap='gray'):
    mi = np.max([im.min(), im.mean() - k * im.std()])
    mx = np.min([im.max(), im.mean() + k * im.std()])
    if len(im.shape) == 3:
        plt.imshow(im, vmin=mi, vmax=mx)
    else:
        plt.imshow(im, vmin=mi, vmax=mx, cmap=cmap,interpolation='bicubic')

    return

#用直方图选择有效参考点（即最大概率的位移），并计算位移量; 
#select effective points with hitogram, and calculate displacement
def mode_model(data, residual_threshold=5):
    (src, dst)=data
    s=src-dst
    r=np.zeros(2)
    D=np.ones((src.shape[1],src.shape[0])).astype('bool') #有效参考点矩阵; position for effective points
    for j in range(2): #循环两个坐标; loop for x,y coordinates
        dat=s[:,j]
        tmp=((np.abs(dat))>0.01) #如果位移在0.01以下，就不考虑。 neglect small displacements
        D[j]*=tmp
        me=0 #初始位移; original displacement
        gain=0.1 #直方图比例尺。0.1代表直方图精度; histogram scale
        rang=[-300,300] #直方图位置计算范围; histogram range
        bins=int((rang[1]-rang[0])/gain+1) #直方图bin
        wd=int(residual_threshold/gain) #直方图峰值宽度; FWHM
        if D[j].sum()>0:
           z=np.histogram(dat,bins=bins,range=rang) #计算直方图; make histogram
           k=z[0].argmax() #直方图最大点; maximum
           x=z[1][k-wd:k+wd+1]
           y=z[0][k-wd:k+wd+1]
           #ymin=y.min()
           # y=np.maximum(y-ymin,0)
           me=np.sum(x*y)/np.sum(y) #在限定宽度内取直方图峰值重心; weight of the histogram, within thr range
        r[j]=me
        D[j]=np.abs(dat-me)<residual_threshold #得到有效参考点矩阵; get the position for effective points
        
    D=D[0]&D[1] #得到XY两个方向共有的参考点; We process for two different directions
    r=np.median(s[D,:],axis=0) #计算有效参考点的中值; calculate the mean displacement for the effective points 
    return -r,D #返回位移量和有效参考点矩阵;  

def all_align(im,align,channels,target=0):  #三个波段彼此求偏移量，并最小二乘求解。

    M=[]
    xy=[]
    for k in range(len(align)): #三个波段循环求偏移量
        L=np.zeros(channels)
        n,m=align[k]
        L[n]=1
        L[m]=-1
       
        im1=im[m]
        im2=im[n]
        d,model,flag,flow=align_opflow(im1,im2,winsize=21,step=2,r_t=2,arrow=0)
#        subfile2=os.path.basename(filelistall[m][i])
        xy.append(d*flag)
        M.append(L*flag)
        print(n,'->',m)
    
    xy=np.array(xy)
    
    #######最小二乘求解
    M=np.array(M)
    M[:,target]=0

    tmp = np.linalg.lstsq(M,xy,rcond=None )
    x=tmp[0]

    print('\033[7;31m RES: \033[0m  ',np.abs(np.dot(M,x)-xy).max(axis=0),np.abs(np.dot(M,x)-xy).mean(axis=0))
    for i in range(1,channels):
        im[i]=immove2(im[i],-x[i,0],-x[i,1])
#    im[2]=immove2(im[2],-x[2,0],-x[2,1])
 
    im=np.array(im)
    return im

def showcolor(K,im,dis=None):
    if K==0:
        dis=plt.imshow(im)
    else:
        dis.set_data(im)
    plt.title(str(K))
    plt.pause(0.1)
    plt.show()
    return  dis 

def basename(filepath):
    import os
    return os.path.splitext(os.path.basename(filepath))[0]

def xcorrcenter(standimage, compimage, R0=2, flag=0):
    import scipy.fftpack as fft
    
    # if flag==1,standimage 是FFT以后的图像，这是为了简化整数象元迭代的运算量。直接输入FFT以后的结果，不用每次都重复计算
    try:
        M, N = standimage.shape

        standimage = zscore2(standimage)
        s = fft.fft2(standimage)

        compimage = zscore2(compimage)
        c = np.fft.ifft2(compimage)

        sc = s * c
        im = np.abs(fft.fftshift(fft.ifft2(sc)))  # /(M*N-1);%./(1+w1.^2);
        cor = im.max()
        if cor == 0:
            return 0, 0, 0

        M0, N0 = np.where(im == cor)
        m, n = M0[0], N0[0]

        if flag:
            m -= M / 2
            n -= N / 2
            # 判断图像尺寸的奇偶
            if np.mod(M, 2): m += 0.5
            if np.mod(N, 2): n += 0.5

            return m, n, cor
        # 求顶点周围区域的最小值
        immin = im[(m - R0):(m + R0 + 1), (n - R0):(n + R0 + 1)].min()
        # 减去最小值
        im = np.maximum(im - immin, 0)
        # 计算重心
        x, y = np.mgrid[:M, :N]
        area = im.sum()
        m = (np.double(im) * x).sum() / area
        n = (np.double(im) * y).sum() / area
        # 归算到原始图像
        m -= M / 2
        n -= N / 2
        # 判断图像尺寸的奇偶
        if np.mod(M, 2): m += 0.5
        if np.mod(N, 2): n += 0.5
    except:
        print('Err in align_Subpix routine!')
        m, n, cor = 0, 0, 0
    return m, n, cor

def fit_dxy(x,xxc,yyc,debug=0): ####边缘和日珥总体位移

    jump=0 #径向抖动的噪声阈值。大于这个阈值的可认为是系统性的. 如果没有明显跳跃JUMP=0，    
    if jump>0:
#        dy=fix_yyc(yyc,G=jump)
        dy=yyc
    else:
#       x=np.array(range(yyc.shape[0]))
        dy=ransacfit(x,yyc,deg=2,r_t=5,debug=debug)[0] ###拟合边缘总体位移
        dx=ransacfit(x,xxc,deg=2,r_t=5,debug=debug)[0] ###拟合日珥位移

    return dx,dy

from scipy.stats import pearsonr as pr
def ransacfit(x,y,deg=2,r_t=3,center=0,debug=0): #用RANSCA方法稳健拟合曲线
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    ran=RANSACRegressor(LinearRegression(),random_state=0,
                        max_trials=100,#最大迭代次数
                        min_samples=3,#最小抽取的内点样本数量
                        residual_threshold=r_t#与拟合曲线距离小于该阈值的是内点，加入到下一轮训练集中，与具体问题有关，目前有能够自动选出适宜的内点阈值方法
                        )
    x=x.reshape(-1,1)
    y=y.reshape(-1,1)
    quadratic = PolynomialFeatures(degree=deg)

    x2 = quadratic.fit_transform(x)


    model=ran.fit(x2,y) #得到模型
    Y=model.predict(x2) #预测
    score=pr(y.flatten().astype('float32'),Y.flatten())[0] #拟合得分
    D=model.inlier_mask_
    if debug==1: print(D.sum(),x.shape[0],round(D.sum()/x.shape[0],3),(Y[D]-y[D]).mean(),(Y[D]-y[D]).std())
    xc,yc=0,0 #日心坐标
    if center==1:
        p1=np.polyfit(x[D,0],y[D,0],deg)
    
        if deg==2:
            a,b,c=p1
            xc=-b/(2*a)
            yc=(4*a*c-b*b+1)/(4*a)
#            print(xc,yc)
        else:
            xc=0
            yc=0
        xc,yc,score=fitcircle(x[D,0],y[D,0],xc,yc)    
      
    return Y ,model,round(score,3),xc,yc

def fitcircle(x,y,xc,yc):

    from scipy  import optimize

   
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2(c):
        global Rsun
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri -Rsun# Ri.mean()
    
    center_estimate = xc, yc
    center_2, ier = optimize.leastsq(f_2, center_estimate)
    
    xc_2, yc_2 = center_2
    yf=np.sqrt(Rsun**2-(x-xc_2)**2)+yc_2

    score=pr(y,yf)[0]

#    print(xc_2,yc_2,score)
    return xc_2,yc_2,score