# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 07:39:15 2022

@author: junse
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

f=open('계산 소요 시간 데이터.txt', 'r')
L=f.readlines()
f.close()

data = np.zeros((5, 3))
for i in range(2, 7):
    tmp = L[i].split()
    for j in range(1, 4):
        data[i-2,j-1]= float(tmp[j])

plt.clf()
font = {'family' : 'normal',        
        'size'   : 25}

plt.rc('font', **font)

thick=1.5
fig,ax=plt.subplots(figsize = (10, 6.3))
plt.xlabel("Img size [pixels]")
plt.ylabel('Process time per\nimage [ms]')



#scatter 그리기
dotsize =30

width =0.1
6




ax.tick_params(axis="y",direction="in", pad=10,length=8,width=thick)
ax.tick_params(axis="x",direction="in", pad=10,length=8,width=thick)
ax.tick_params(axis="y",which="minor",direction="in", pad=10,length=5,width=thick)
ax.tick_params(axis="x",which="minor",direction="in", pad=10,length=5,width=thick)
ax.xaxis.set_minor_locator(MultipleLocator(1000))
ax.yaxis.set_minor_locator(MultipleLocator(5))

ax.spines["top"].set_linewidth(thick)
ax.spines["bottom"].set_linewidth(thick)
ax.spines["left"].set_linewidth(thick)
ax.spines["right"].set_linewidth(thick)


ax.tick_params(axis="y",direction="in", pad=10)
ax.tick_params(axis="x",direction="in", pad=10)

plt.xlim(1000,5000)
plt.ylim(0, 20)
plt.xticks(np.arange(1000, 5001,1000))
plt.yticks(np.arange(0, 21,5))

image_sizes = np.arange(1000, 5001, 1000)
ax.plot(image_sizes, data[:,0], 'o-', color = '#6868AC')
ax.plot(image_sizes, data[:,1], 'o-', color = '#E9435E')
ax.plot(image_sizes, data[:,2], 'o-', color = '#ECC371')

fig.savefig('find_nonzero_time.png', bbox_inches='tight', dpi = 300)






