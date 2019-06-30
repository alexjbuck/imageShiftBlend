#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider
import argparse
from PIL import Image

import time
global t0,w
t0 = time.time()

def printStatus(level,text):
	""" 
	Take a bit of text and prepend -'s and a > based off the depth.
	Used for status messages, with depth denoting a heirarchical relationship.
	"""
	text=str(text)
	global t0
	pre = "[{0:>7.2f}] ".format(time.time()-t0)
	for x in range(0,level):
		pre += "-"
	pre += "> "
	print(pre+text)

def XYW(c,n,extra,w):
	"""This function calculates values for X, Y, and V, the weights
	given values of c (scale), n (number of frames), extra (buffer frames)
	w is the width in pixels of each frame.
	"""
	n = int(n)
	w = int(w)
	extra = int(extra)
	# N linspace from 0 to n with integer spacing, length n-1
	N = np.linspace(-extra,n+extra,2*extra+n-1)
	# X linspace from 0 to n with w points representing the w pixels wide of the frame
	X = np.linspace(-extra,n+extra,w,endpoint=True)
	# Y is dimension n x w
	Y = np.zeros((len(N)+1,len(X)))
	# W is dimension n x w
	W = np.zeros((len(N)+1,len(X)))

	# Calculate Y, the boundary lines for determining weights
	Y[0] = np.ones(len(X))
	for j in range(0,len(N)):
		Y[j+1] = 1/2 + np.arctan(c*(X-N[j]))/math.pi

	# Calculate W, the weights
	for j in range(0,len(N)):
		W[j] = Y[j]-Y[j+1]
	W[len(N)] = Y[len(N)]-np.zeros(len(X))

	# Return X, Y, and W
	return X,Y,W

def replot(fig,ax,n,X,Y,W):
	"""This function clears the subplot axes and then replots
	"""
	ax[0].clear()
	ax[0].plot(X,Y.transpose())
	ax[1].clear()
	ax[1].plot(X,W.transpose())
	ax[0].set_xlim(0,n)
	ax[1].set_xlim(0,n)
	ax[0].set_title('Image Transitions')
	ax[1].set_title('Image Blending Weights')
	fig.canvas.draw()
	fig.canvas.flush_events()

def update(val):
	"""This function updates the values of c, n, and e then runs XYW
	and updates the plots"""
	global fig, ax, w
	c = sc.val
	n = sn.val
	e = se.val
	w = w
	X,Y,W = XYW(c,n,e,w)
	replot(fig,ax,n,X,Y,W)
	filename="img_"+str(n)+"_"+str(c)+".jpg"
	outImg = Image.fromarray(reweight(frames,W).astype('uint8'))
	outImg.show()
	outImg.save(filename)

def reweight(frames,W):
	printStatus(0,"Beginning frame weighting")
	outFrame = np.zeros_like(frames[0])
	for i in range(0,frames.shape[0]):
		printStatus(1,"Weighting frame "+str(i))
		# iterate over each frame within frames
		for j in range(0,frames.shape[2]):
			# iterate over every pixel column of the frame (shape[1] is the row index)
			# multiply every pixel in the column by the weight for this image (i) and this column (j)
			# Add the scaled frame to the outFrame
			outFrame[:,j,:]+=frames[i,:,j,:]*W[i,j]
	printStatus(1,"Done!")
	return outFrame

parser = argparse.ArgumentParser(description='Input N image files and blend them left to right using arctan based weighting. N-2 central frames with weighting peaks and 2 edge frames.')
parser.add_argument('--input','-i',help="Input files.",required=True,dest='imgList',nargs='+',type=str,default=('1.jpg','2.jpg','3.jpg'))

args=parser.parse_args()
imgList=args.imgList

printStatus(0,imgList)

n = int(len(imgList))
frame = np.asarray(Image.open(imgList[0]));
h,w = frame.shape[0],frame.shape[1]

frames = np.zeros((n,frame.shape[0],frame.shape[1],frame.shape[2]))
for i in range(0,n):
	frames[i] = np.asarray(Image.open(imgList[i]).copy()).astype('uint32')
	printStatus(1,imgList[i]+" loaded.")

c = 2.7
n = int(n)
e = int(0)
w = int(w)
X,Y,W = XYW(c,n,e,w)
filename="img_"+str(n)+"_"+str(c)+".jpg"
outImg = Image.fromarray(reweight(frames,W).astype('uint8'))
outImg.show()
outImg.save(filename)

fig,ax = plt.subplots(2,constrained_layout=True)

fig2 = plt.figure(2)

axcolor = 'pink'
axc = plt.axes([0.15, 0.05, 0.7, 0.3], facecolor=axcolor)
axn = plt.axes([0.15, 0.35, 0.7, 0.3], facecolor=axcolor)
axe = plt.axes([0.15, 0.65, 0.7, 0.3], facecolor=axcolor)

sc = Slider(axc, 'c', .1, 10.0, valinit=c, valstep=.1)
sn = Slider(axn, 'n', 1, 10, valinit=n, valstep=1)
se = Slider(axe, 'e', 0, 10, valinit=e,valstep=1)

sc.on_changed(update)
sn.on_changed(update)
se.on_changed(update)

replot(fig,ax,n,X,Y,W)

plt.show()