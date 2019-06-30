#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Slider

def XYW(c,n,extra):
	"""This function calculates values for X, Y, and W, the weights
	given values of c (scale), n (number of frames), extra (buffer frames)"""
	n = int(n)
	extra = int(extra)
	N = np.linspace(-extra,n+extra,2*extra+n+1)
	X = np.linspace(-extra,n+extra,256*n,endpoint=True)
	Y = np.zeros((len(N),len(X)))
	W = np.zeros((len(N)+1,len(X)))

	# Calculate Y, the boundary lines for determining weights
	for j in range(0,len(N)):
		Y[j] = 1/2 + np.arctan(c*(X-N[j]))/math.pi

	# Calculate W, the weights
	W[0] = np.ones(len(X))-Y[0]
	for j in range(1,len(N)):
		W[j] = Y[j-1]-Y[j]
	W[len(N)] = Y[len(N)-1]-np.zeros(len(X))

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
	"""This function updates the values of c, n, and e then runs XYV 
	and updates the plots"""
	global fig, ax
	c = sc.val
	n = sn.val
	e = se.val
	X,Y,w = XYW(c,n,e)
	replot(fig,ax,n,X,Y,W)

c = 5
n = int(5)
e = int(0)

X,Y,W = XYW(c,n,e)

fig,ax = plt.subplots(2,constrained_layout=True)

fig2 = plt.figure(2)

axcolor = 'pink'
axc = plt.axes([0.15, 0.05, 0.7, 0.3], facecolor=axcolor)
axn = plt.axes([0.15, 0.35, 0.7, 0.3], facecolor=axcolor)
axe = plt.axes([0.15, 0.65, 0.7, 0.3], facecolor=axcolor)

sc = Slider(axc, 'c', .1, 100.0, valinit=5, valstep=.1)
sn = Slider(axn, 'n', 1, 10, valinit=4, valstep=1)
se = Slider(axe, 'e', 0, 10, valinit=0,valstep=1)

sc.on_changed(update)
sn.on_changed(update)
se.on_changed(update)

replot(fig,ax,n,X,Y,W)

# plt.figure(3)
# S = np.zeros_like(X)
# for j in range(0,len(N)+1):
# 	S += V[j]
# plt.plot(X,S)

plt.show()