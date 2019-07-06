#!/usr/local/bin/python3
from matplotlib.widgets import Slider
from PIL import Image, ImageMath
import colorsys, argparse, math, time, numpy as np, matplotlib.pyplot as plt

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
def linearWeights(b,n,e,w):
	'''
	Currently outputs all zero weights. Placeholder for further development
	'''
	N = np.linspace(-e,n+e,2*e+n-1)
	X = np.linspace(-e,n+e,w,endpoint=True)
	Y = np.zeros((len(N)+1,len(X)))
	W = np.zeros((len(N)+1,len(X)))
	return X,Y,W
def logisticWeights(b,n,e,w):
	'''
	Currently outputs all zero weights. Placeholder for further development
	'''
	N = np.linspace(-e,n+e,2*e+n-1)
	X = np.linspace(-e,n+e,w,endpoint=True)
	Y = np.zeros((len(N)+1,len(X)))
	W = np.zeros((len(N)+1,len(X)))
def arctanWeights(b,n,e,w):
	"""
	This function calculates values for X, Y, and V, the weights
	given values of b (blur factor), n (number of frames), e (buffer frames)
	w is the width in pixels of each frame
	"""
	n = int(n)
	w = int(w)
	e = int(e)
	N = np.linspace(-e,n+e,2*e+n-1)	# N linspace from 0 to n with integer spacing, length n-1
	X = np.linspace(0,n,w,endpoint=True)	# X linspace from 0 to n with w points representing the w pixels wide of the frame
	Y = np.zeros((len(N)+1,len(X)))	# Y is dimension n x w
	W = np.zeros((len(N)+1,len(X)))	# W is dimension n x w

	# Calculate Y, the boundary lines for determining weights
	Y[0] = np.ones(len(X))
	for j in range(0,len(N)):
		Y[j+1] = 1/2 + np.arctan(b*(X-N[j]))/math.pi

	# Calculate W, the weights
	for j in range(0,len(N)):
		W[j] = Y[j]-Y[j+1]
	W[len(N)] = Y[len(N)]-np.zeros(len(X))

	# Return X, Y, and W (X and Y for plotting)
	return X,Y,W
def replot(fig,axi,axw,b,X,W,frame):
	"""This function clears the subplot axes and then replots
	"""
	axi.clear()
	axi.imshow(frameCMtoRGB(frame))
	axi.set_title(args.cm + ' - Image Blend Factor: '+str(b))
	axw.clear()
	axw.plot(X,W.transpose())
	axw.set_title('Image Blending Weights')
	fig.canvas.draw()
	fig.canvas.flush_events()
def update(val):
	"""This function updates the values of b, n, and e then runs calcWeights
	and updates the plots"""
	b = round(sb.val,1)
	X,Y,W = calcWeights(b,n,e,w)
	frame = reweight(frames,W)
	replot(fig,axi,axw,b,X,W,frame)

def rows(frame):
	return frame.shape[0]
def columns(frame):
	return frame.shape[1]
def depth(frame):
	return frame.shape[2]

def rgbReweight(frames,W):
	printStatus(0,"Beginning frame weighting")
	outFrame = np.zeros_like(frames[0])
	for i in range(0,frames.shape[0]): # iterate over each frame within frames
		printStatus(1,"Weighting frame "+str(i))
		for j in range(columns(outFrame)): # iterate over every column of the frame
			# multiply every pixel in the column by the weight for this image (i) and this column (j)
			# Add the scaled frame to the outFrame
			outFrame[:,j,:]+=frames[i,:,j,:]*W[i,j]
	printStatus(1,"Done!")
	return outFrame

def hsvReweight(frames,W):
	printStatus(0,"Beginning frame weighting")
	outFrame = np.zeros_like(frames[0])
	for i in range(0,frames.shape[0]): # iterate over each frame within frames
		printStatus(1,"Weighting frame "+str(i))
		for j in range(columns(outFrame)): # iterate over every column of the frame (shape[1] is the row index)
			# multiply every pixel in the column by the weight for this image (i) and this column (j)
			# Add the scaled frame to the outFrame
			outFrame[:,j,:]+=frames[i,:,j,:]*W[i,j] # rx3 * 1x1
	printStatus(1,"Done!")
	return outFrame

def hsvReweight2(frames,W):
	printStatus(0,"Beginning frame weighting")
	outFrame = np.zeros_like(frames[0])
	for k in range(columns(outFrame)): # iterate over every column of the frame (shape[1] is the row index)
		printStatus(1,"Weighting column "+str(k))
		for j in range(rows(outFrame)): # iterate over every row of the frame
			printStatus(2,"Weighting row "+str(j))
			ss = 0
			vs = 0
			h = np.zeros(frames.shape[0])
			s = np.zeros(frames.shape[0])
			v = np.zeros(frames.shape[0])
			x = np.zeros(frames.shape[0])
			y = np.zeros(frames.shape[0])
			for i in range(0,frames.shape[0]): # iterate over each frame within frames
				h[i],s[i],v[i] = frames[i,j,k,:]
				x[i] = math.cos(h[i]*2*math.pi)*W[i,k]
				y[i] = math.cos(h[i]*2*math.pi)*W[i,k]
				ss += s[i]*W[i,k]
				vs += v[i]*W[i,k]
			hs = math.atan2(sum(y),sum(x))/(2*math.pi)
			outFrame[j,k,:] = (hs,ss,vs)
	printStatus(1,"Done!")
	return outFrame


def frameRGBtoRGB(frame):
	return frame.astype('uint8')
def frameRGBtoHSV(frame):
	frame = frame.astype('float')
	for i in range(frame.shape[0]):
		for j in range(frame.shape[1]):
			r,g,b = frame[i,j]
			r /= 255.0
			g /= 255.0
			b /= 255.0
			h,s,v = colorsys.rgb_to_hsv(r,g,b)
			frame[i,j] = (h,s,v)
	return frame
def frameHSVtoRGB(frame):
	for i in range(frame.shape[0]):
		for j in range(frame.shape[1]):
			h,s,v = frame[i,j]
			h %= 1
			s = min(s,1)
			r,g,b = colorsys.hsv_to_rgb(h,s,v)
			r *= 255
			g *= 255
			b *= 255
			frame[i,j] = (r,g,b)
	np.clip(frame,0,255,out=frame)
	return frame.astype('uint8')
def frameRGBtoHSL(frame):
	frame = frame.astype('float')
	for i in range(frame.shape[0]):
		for j in range(frame.shape[1]):
			r,g,b = frame[i,j]
			r /= 255.0
			g /= 255.0
			b /= 255.0
			h,l,s = colorsys.rgb_to_hls(r,g,b)
			frame[i,j] = (h,s,l)
	# return frame
def frameHSLtoRGB(frame):
	for i in range(frame.shape[0]):
		for j in range(frame.shape[1]):
			h,s,l = frame[i,j]
			# h,s = (math.atan2(y,x)/(2*math.pi),math.sqrt(x**2+y**2))
			h %= 1
			s = min(s,1)
			r,g,b = colorsys.hls_to_rgb(h,l,s)
			r *= 255
			g *= 255
			b *= 255
			frame[i,j] = (r,g,b)
	np.clip(frame,0,255,out=frame)
	return frame.astype('uint8')
def horizImageMask(size,W): # THIS IS FUTURE WORK AREA, MAY NOT PURSUE
	return np.broadcast_to(W,size)
def maskImage(frame,mask): # THIS IS FUTURE WORK AREA, MAY NOT PURSUE
	return ImageMath.eval("a*b",a=frame,b=mask)

"""
	Argument Parser
"""
parser = argparse.ArgumentParser(description='Input N image files and blend them left to right using arctan based weighting. N-2 central frames with weighting peaks and 2 edge frames.')
parser.add_argument('-i',help="Input files.",required=False,dest='imgList',nargs='+',type=str,default=('0.jpg','1.jpg','2.jpg','3.jpg','4.jpg'))
parser.add_argument('--interactive',help="Toggles on interactive mode.",required=False,dest='interactive',action='store_true',default=False)
parser.add_argument('--thumbsize',help="Sets bounding box for thumbnail image size.",nargs=2,type=int,required=False,default=(480,480))
parser.add_argument('--outsize',help="Sets bounding box for output image size.",nargs=2,type=int,required=False,default=None)
parser.add_argument('-b',help='Assigns the blend factor used to determine the weights. Simply it sets the horizontal scaling of the arctan functions used to generate weights.',required=False,dest='b',nargs=1,default=2.7,type=float)
parser.add_argument('-wf',help='Assign which weighting function to use for blending the image.',required=False,default='arctan',choices=['arctan','linear','logistic'])
parser.add_argument('-cm',help='Assign which color model to use when blending images.',required=False,default='hsl',choices=['rgb','hsl','hsv'])

args 		= parser.parse_args()
imgList		= args.imgList
b 			= args.b
interactive = args.interactive
outsize 	= args.outsize
"""
	Done Parsing Arguments
"""

"""
	Begin Main Program
"""
# Select weighting function
if args.wf=="arctan":
	calcWeights = arctanWeights
elif args.wf=="linear":
	calcWeights	= linearWeights
elif args.wf=="logistic":
	calcWeights	= logisticWeights
else:
	print("Something went very wrong when parsing weighting function arguments. I'm going to give up.")
	quit()

# Select color model conversion function
if args.cm=="rgb":
	frameRGBtoCM = frameRGBtoRGB
	frameCMtoRGB = frameRGBtoRGB
	reweight = rgbReweight
elif args.cm=="hsl":
	frameRGBtoCM = frameRGBtoHSL
	frameCMtoRGB = frameHSLtoRGB
elif args.cm=="hsv":
	frameRGBtoCM = frameRGBtoHSV
	frameCMtoRGB = frameHSVtoRGB
	reweight = hsvReweight
else:
	print("Something went very wrong when parsing color model arguments. I'm going to give up.")
	quit()

# Open first image, resize as needed
img = Image.open(imgList[0])
if 	interactive:
	thumbsize = args.thumbsize
else:
	if outsize==None:
		thumbsize = img.size
	else:
		thumbsize = outsize

img.thumbnail(thumbsize)

# Pull image dimensions from first loaded image, initialize frames ndarray
n 		= int(len(imgList))
frame  	= np.asarray(img);
h,w 	= frame.shape[0:2]
frames 	= np.zeros((n,h,w,frame.shape[2]))

# Begin loading the images to memory in 'frames'
printStatus(0,'Loading Images: '+ str(imgList))
for i in range(0,n):
	img = Image.open(imgList[i])
	img.thumbnail(thumbsize)
	frames[i] = frameRGBtoCM(np.asarray(img))
	printStatus(1,imgList[i]+' loaded.')
printStatus(1,'Done!')

# Fixing parameter values as required
e = int(0) # Forcing to zero until I implment how to handle this on the image. Allowing extra > 0 means the weights for the image frame no longer start at the zero index and I haven't implemented a way to handle this yet.

if interactive:
	fig = plt.figure()
	gs 	= fig.add_gridspec(10,1)
	axi = fig.add_subplot(gs[0:6])
	axw = fig.add_subplot(gs[6:9])
	axb = fig.add_subplot(gs[9])
	axw.set_ylim(0,1)

	sb = Slider(axb, 'b', .1, 10.0, valinit=b, valstep=.1)
	sb.on_changed(update)
	update(b)
	plt.show()
else:
	W = calcWeights(b,n,e,w)[2]
	outFrame = frameCMtoRGB(reweight(frames,W))
	filename = args.cm+str(n)+"_"+str(b)+".jpg"
	outImg = Image.fromarray(outFrame).save(filename)
	printStatus(0,'File saved: '+filename)


# 