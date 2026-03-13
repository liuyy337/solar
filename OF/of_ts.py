# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:31:08 2024

@author: xuyang
"""

######Basic Optical Flow alignment code for time-series imaging dataset
from matplotlib import pyplot as plt
import numpy as np
import glob,os
import cv2
import tools_of as tools
plt.close('all')

dir_in=r'F:/gst_ss_2024/data/20220517/vissr/sr/r080'
dir_in=r'F:/gst_ss_2024/data/20200513/vissr/b080'
dir_in=r'F:/gst_ss_2024/data/of_ts_test1'
# dir_out=os.path.dirname(dir_in)+'/of'
dir_out=dir_in+'/of/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
filelist=sorted(glob.glob(dir_in+'/*.fts'))
tot=len(filelist)
nr=0#reference frame, 0==>nr
im0,h0=tools.fitsread(filelist[nr])
OT = h0['TIME-OBS']
T1 = tools.num_time(OT)

h,w=im0.shape
icube=np.zeros((tot,h,w))
icube=np.zeros((tot,1800,2100))
for ii in range(tot):
    im,hd2= tools.fitsread(filelist[ii])
    OT = hd2['TIME-OBS']
    T2 = tools.num_time(OT)
    difT=T2-T1#+210*15#time difference to the reference frame, to do derotation
#    print(difT)
    rot = 360./24./3600.*difT  
    im = tools.imrotate(im,rot)#derotate
    print(filelist[ii])
    icube[ii,:,:]=im#[250:2050,200:2300]
    
ncube=tools.cubealign(icube.copy(),wd=100,winsize=31)
ncube=ncube.astype(np.float32)
tcube=[]
fn=[]
print('write fits')
for jj in range(tot):
    tcube.append(os.path.basename(filelist[jj]))
    nfn=dir_out+os.path.basename(filelist[jj])
    im,head= tools.fitsread(filelist[ii])
    tools.fitswrite(nfn,ncube[jj,:,:],head)   
    print(nfn)
    
tools.array2movie(ncube,movie_name=dir_out+'of_moive',title_cube=tcube)