import numpy
import cv2

# this exercise references "Pyramid Methods in Image Processing" by Adelson et al.

numpyFirst = cv2.imread(filename='./samples/multiband-apple.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
numpySecond = cv2.imread(filename='./samples/multiband-orange.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# blend the apple and the orange using multiband blending with paplacian pyramids

# creating a laplacian pyramid with seven levels for each of the two images

numpyFirst = [ numpyFirst ]
numpySecond = [ numpySecond ]

for intLevel in range(6):
	numpyFirst.append(cv2.pyrDown(numpyFirst[-1]))
	numpySecond.append(cv2.pyrDown(numpySecond[-1]))

	numpyFirst[-2] -= cv2.pyrUp(numpyFirst[-1])
	numpySecond[-2] -= cv2.pyrUp(numpySecond[-1])
# end

# combine the two laplacian pyramids and create a new laplacian pyramid to blend the two images
# specifically, take the left half from numpyFirst and the right half from numpySecond at each level
# afterwards, collapse numpyPyramid to obtain the blended result and store it in numpyOutput
# print(len(numpyFirst), len(numpySecond))
numpyPyramid = []
for left,right in zip(numpyFirst,numpySecond):
	rows,cols,dpt = left.shape
	mixed = numpy.hstack((left[:,0:int(cols/2)], right[:,int(cols/2):]))
	numpyPyramid.append(mixed)

numpyOutput = numpyPyramid[6]
for i in reversed(range(6)):
	# print(numpyOutput.shape)
	numpyOutput = cv2.pyrUp(numpyOutput)
	numpyOutput = cv2.add(numpyOutput, numpyPyramid[i])

cv2.imwrite(filename='./10-multiband.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))