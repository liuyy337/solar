# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:12:18 2024

@author: xuyang
"""

######Optical flow alignment code for spectroscopic imageset.
######


from matplotlib import pyplot as plt
import numpy as np
import os,glob
import tools_of as tools

plt.close('all')
dir_in='F:/gst_ss_2024/data/of_sp_test1/'
def sp_of(dir_in):
    dir_out=dir_in+'of_sp/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    wave=['r120','r100','r080','r060','r040','r020','r000','b020','b040','b060','b080','b100','b120']
    align=[[1,0],[2,1],[3,2],[4,3],[5,4],[6,5],[7,6],[8,7],[9,8],[10,9],[11,10],[12,11],
           [0,12],[1,11],[2,10],
           [0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[1,3],[3,5],[5,7],[7,9],[9,11]]
    wavenum=len(wave)
    filelist=[]
    for i in range(wavenum):
        tmpfile=sorted(glob.glob(dir_in+'/*'+wave[i]+'*.fts'))
        filelist.append(tmpfile)
        
    im=[]
    for j in range(wavenum):
        tmp=tools.fitsread(filelist[j][0])[0]
    #        tmp=tmp/tmp[wd:-wd,wd:-wd].mean()*10000
        im.append(tmp)
        print(filelist[j][0])
    im=np.array(im)
    
    im=tools.all_align(im,align,wavenum)
    im=im.astype(np.float32) 
    subfile=os.path.basename(filelist[6][0])
    tools.fitswrite(dir_out+subfile,im,header=None) 

sp_of(dir_in)