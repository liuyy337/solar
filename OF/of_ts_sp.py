# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:48:45 2024

@author: xuyang
"""

######Alignment for time-series spectroscopic imaging dataset
######1.Time-Series alignment on one target wavelength; 2.Spectroscpic alignment to the target wavelength at each moment
from matplotlib import pyplot as plt
import numpy as np
import os,glob
import tools_of as tools


      
dir_in='D:/Doppler/data/20211030/sr'
dir_in='G:/gst_ss_2024/data/of_sp_test1'

def ts_sp(dir_in):
    dir_out=os.path.dirname(dir_in)+'/tssp/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    ###Should make this part automatic    
    # wave=['r120','r100','r080','r060','r040','r020','r000','b020','b040','b060','b080','b100','b120']
    # align=[[1,0],[2,1],[3,2],[4,3],[5,4],[6,5],[7,6],[8,7],[9,8],[10,9],[11,10],[12,11],
    #        [0,12],[1,11],[2,10],
    #        [0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[1,3],[3,5],[5,7],[7,9],[9,11]]
    
    wave=['r080','r060','r040','r000','b040','b060','b080']
    align=[[1,0],[2,1],[3,2],[4,3],[5,4],[6,5],
           [0,6],[1,5],[2,4],
           [0,2],[2,4],[4,6],[1,3],[3,5]]
    
    plt.close('all')
    wavenum=len(wave)
    wd=500
    target=6#(wavenum+1)/2#r100/r080
    centlist=sorted(glob.glob(dir_in+'/'+wave[target]+'/*.fts'))
    tot=len(centlist)
    im0org,head0=tools.fitsread(centlist[0])
    imm,head0=tools.fitsread(centlist[round(tot/2)]) #序列的一半作为相关的基准点#Reference point, middle of TS
    Tr=tools.num_time(head0['TIME-OBS'])
    im0=im0org.copy()
    K=0
    # dis=None
    
    im1=im0.copy()
    im1=im1/im1[wd:-wd,wd:-wd].mean()*10000#normlize image intensity, quiet area may be better.
    dxy=[]
    xy_shift=[]
    xxc=[]
    yyc=[]
    
    for i in range(tot):###Step1 TS alignment
        im2org,hd=tools.fitsread(centlist[i])
        difT=tools.num_time(hd['TIME-OBS'])-Tr###image derotation
        rot = 360./24/3600*difT            
        im2org = tools.imrotate(im2org,rot)
        im2=im2org/im2org[wd:-wd,wd:-wd].mean()*10000
    
        d,model,flag,flow =tools.align_opflow(im1,im2,winsize=31,step=5,r_t=5,arrow=0)
        dxy.append(d)
        im2dxy=np.sum(np.array(dxy),axis=0) #本帧的累计位移量#total displacement for this frame
        xy_shift.append(im2dxy) #累计位移量序列 
        im2new=tools.immove2(im2org,-im2dxy[0],-im2dxy[1]) #移动本帧到目标图像#correct the displacement(move image)
    
        print(i,-im2dxy)
        im3=im2new[wd:-wd,wd:-wd]
    
        Dx,Dy,cor=tools.xcorrcenter(imm[wd:-wd,wd:-wd],im3) 
        print(i,Dy,Dx,cor)
    
        xxc.append(Dx)
        yyc.append(Dy)
        im1=im2.copy()
        K=K+1   
    xxc=np.array(xxc)
    yyc=np.array(yyc) 
    xy_shift=np.array(xy_shift)   
    x=np.array(range(len(xxc)))
    Dx,Dy=tools.fit_dxy(x,xxc,yyc)
    plt.figure()
    plt.plot(xxc) 
    plt.plot(yyc)
    plt.plot(Dx)
    plt.plot(Dy)
    
    plt.figure('color')
    filelist=[]
    for i in range(wavenum):
        tmpfile=sorted(glob.glob(dir_in+'/'+wave[i]+'/*.fts'))
        filelist.append(tmpfile)
        
    for i in range(0,tot):
    
        im=[]
        for j in range(wavenum):
            tmp=tools.fitsread(filelist[j][i])[0]
    #        tmp=tmp/tmp[wd:-wd,wd:-wd].mean()*10000
            im.append(tmp)
            print(filelist[j][i])
        im=np.array(im)    
        
        for j in range(wavenum):###correct TS displacement for each moment
            im[j]=tools.immove2(im[j],-xy_shift[i,0]+Dy[i][0],-xy_shift[i,1]+Dx[i][0])
        
        im=tools.all_align(im,align,wavenum)###Step2 SP alignment for each momnent
        im=im.astype(np.float32) 
        subfile=os.path.basename(filelist[5][i])
        print(i, subfile)
        tools.fitswrite(dir_out+'/'+subfile,im,header=None)
        
ts_sp(dir_in)
