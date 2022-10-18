from __future__ import division
from matplotlib import pyplot as plt
from matplotlib import colors
import time, re, cv2, numpy as np
from os import listdir
from os.path import isfile, join
from random import randint

debut = time.time()

for i in range (0,43):
	i= "%02d" % (i)
	mypath = '/media/remy/TOURO/Images/000' + i
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	taille=len(onlyfiles)
	#print taille
	dim = 2250-taille
	j=0
	while dim != 0 :
		j=(randint(0, len(onlyfiles)-1))
		img = cv2.imread(join(mypath, onlyfiles[j]))
		image=img
		#flip the image
		k=(randint(0,3))
		if k==0:
            img = img
		elif k==1:
            img =cv2.flip( img, 0 )
		elif k == 2:
            img =cv2.flip( img, 1 )
		elif k == 3:
            img =cv2.flip( img, -1 )
		#cv2.imshow("Original", img)
		#cv2.waitKey(0)
        
		# rotate the image by k degrees
		k=(randint(-15,15))
		(h, w) = img.shape[:2]
		center = (w / 2, h / 2)
		M = cv2.getRotationMatrix2D(center, k, 1.0)
		img = cv2.warpAffine(img, M, (w, h))
		numero = "%05d" % (dim)
		name=mypath+'/'+onlyfiles[j][:11]+'_rev_'+str(numero)
		cv2.imwrite(name+'.ppm', img)
		# cv2.imshow("Original", img)
		# cv2.waitKey(0)
        
		'''''plt.subplot(121), plt.imshow(image, cmap='gray')
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(122), plt.imshow(img, cmap='gray')
		plt.title(name), plt.xticks([]), plt.yticks([])
		plt.show()'''
        
		print i, dim
        
		dim-=1