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
def linearWeights(b,n,extra,w):
	'''
	Currently outputs all zero weights. Placeholder for further development
	'''
	N = np.linspace(-extra,n+extra,2*extra+n-1)
	X = np.linspace(-extra,n+extra,w,endpoint=True)
	Y = np.zeros((len(N)+1,len(X)))
	W = np.zeros((len(N)+1,len(X)))
	return X,Y,W
def arctanWeights(b,n,extra,w):
	"""
	This function calculates values for X, Y, and V, the weights
	given values of b (blur factor), n (number of frames), extra (buffer frames)
	w is the width in pixels of each frame
	"""
	n = int(n)
	w = int(w)
	extra = int(extra)
	# N linspace from 0 to n with integer spacing, length n-1
	N = np.linspace(-extra,n+extra,2*extra+n-1)
	# X linspace from 0 to n with w points representing the w pixels wide of the frame
	X = np.linspace(0,n,w,endpoint=True)
	# Y is dimension n x w
	Y = np.zeros((len(N)+1,len(X)))
	# W is dimension n x w
	W = np.zeros((len(N)+1,len(X)))

	# Calculate Y, the boundary lines for determining weights
	Y[0] = np.ones(len(X))
	for j in range(0,len(N)):
		Y[j+1] = 1/2 + np.arctan(b*(X-N[j]))/math.pi

	# Calculate W, the weights
	for j in range(0,len(N)):
		W[j] = Y[j]-Y[j+1]
	W[len(N)] = Y[len(N)]-np.zeros(len(X))

	# Return X, Y, and W
	return X,Y,W
def replot(fig,axi,axw,b,X,W,frame):
	"""This function clears the subplot axes and then replots
	"""
	axi.clear()
	axi.imshow(frame)
	axi.set_title('Image Blend Factor: '+str(b))

	axw.clear()
	axw.plot(X,W.transpose())
	axw.set_title('Image Blending Weights')
	fig.canvas.draw()
	fig.canvas.flush_events()
def update(val):
	"""This function updates the values of b, n, and e then runs calcWeights
	and updates the plots"""
	b = round(sb.val,1)
	# Removing n and e sliders
	# n = sn.val
	# e = se.val
	X,Y,W = calcWeights(b,n,e,w)
	frame = reweight(frames,W)
	replot(fig,axi,axw,b,X,W,frame)
	# filename="img_"+str(n)+"_"+str(b)+".jpg"
	# Image.fromarray(frame).save(filename)
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
	# Clip values to 0-255
	np.clip(outFrame,0,255,out=outFrame)
	printStatus(1,"Done!")
	return outFrame.astype('uint8')

"""
Argument Parser
"""
parser = argparse.ArgumentParser(description='Input N image files and blend them left to right using arctan based weighting. N-2 central frames with weighting peaks and 2 edge frames.')
parser.add_argument('-i',help="Input files.",required=False,dest='imgList',nargs='+',type=str,default=('0.jpg','1.jpg','2.jpg','3.jpg','4.jpg'))
parser.add_argument('--interactive',help="Toggles on interactive mode.",required=False,dest='interactive',action='store_true',default=False)
parser.add_argument('-b',help='Assigns the blend factor used to determine the weights. Simply it sets the horizontal scaling of the arctan functions used to generate weights.',required=False,dest='b',nargs=1,default=2.7,type=float)
parser.add_argument('-wf',help='Assign which weighting function to use for blending the image.',required=False,default='arctan',choices=['arctan','linear'])
args 	= parser.parse_args()
imgList	= args.imgList
b 		= args.b
interactive = args.interactive

# Select weighting function
if args.wf=="arctan":
	calcWeights = arctanWeights
elif args.wf=="linear":
	calcWeights	= linearWeights
else:
	print("Something went very wrong when parsing arguments. I'm going to give up.")
	quit()

"""
Done Parsing Arguments

"""

'''
	Begin Main Program
'''
img = Image.open(imgList[0]).copy()
if 	interactive:
	thumbsize = 720,720
else:
	thumbsize = img.size

printStatus(0,imgList)
n = int(len(imgList))
img.thumbnail(thumbsize)
frame = np.asarray(img);
frames = np.zeros((n,frame.shape[0],frame.shape[1],frame.shape[2]))
h,w = frame.shape[0],frame.shape[1]

# Begin loading the images to memory
printStatus(0,'Loading Images...')
for i in range(0,n):
	img = Image.open(imgList[i]).copy()
	img.thumbnail(thumbsize)
	frames[i] = np.asarray(img)
	printStatus(1,imgList[i]+' loaded.')
printStatus(1,'Done!')

# Initial parameter values
n = int(n)
e = int(0) # Forcing to zero until I implment how to handle this on the image. Allowing extra > 0 means the weights for the image frame no longer start at the zero index and I haven't implemented a way to handle this yet.
w = int(w)

if interactive:
	fig = plt.figure(constrained_layout=True)
	gs 	= fig.add_gridspec(10,1)
	axi = fig.add_subplot(gs[0:5])
	axw = fig.add_subplot(gs[6:8])
	axb = fig.add_subplot(gs[9])
	# fig,(axi,axw,axc) = plt.subplots(3,1)

	# n slider isn't needed when working with real image set, n is provided by number of images assuming extra is zero
	# axn = plt.axes([0.15, 0.35, 0.7, 0.3], facecolor=axcolor)
	# Removing extra feature. Only allowing extra=0 for now.
	# axe = plt.axes([0.15, 0.65, 0.7, 0.3], facecolor=axcolor)

	axw.set_ylim(0,1)

	sb = Slider(axb, 'b', .1, 10.0, valinit=b, valstep=.1)
	# sn = Slider(axn, 'n', 1, 10, valinit=n, valstep=1)
	# se = Slider(axe, 'e', 0, 10, valinit=e,valstep=1)

	sb.on_changed(update)
	# sn.on_changed(update)
	# se.on_changed(update)

	update(b)
	# replot(fig,ax,n,X,Y,W)

	plt.show()

else:
	W = calcWeights(b,n,e,w)[2]
	outFrame = reweight(frames,W)
	filename="out_"+str(n)+"_"+str(b)+".jpg"
	Image.fromarray(outFrame).save(filename)
	printStatus(0,'File saved: '+filename)


