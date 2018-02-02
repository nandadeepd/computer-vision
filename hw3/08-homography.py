import numpy
import cv2
import math, tqdm

numpyInput = cv2.imread(filename='./samples/homography-2.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# estimate the homography matrix between matching points and warp the image using bilinear interpolation

# creating the mapping between the four corresponding points

intSrc = [ [266, 343], [646, 229], [388, 544], [777, 538] ]
intDst = [ [302, 222], [746, 231], [296, 490], [754, 485] ]

# construct the linear homogeneous system of equations
# use a singular value decomposition to solve the system
# in practice, cv2.findHomography can be used for this
# however, do not use this function for this exercise


A = []

for intCorr in range(len(intSrc)):
	sx, sy = intSrc[intCorr][0], intSrc[intCorr][1]
	dx, dy = intDst[intCorr][0], intDst[intCorr][1]

	A.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx, -dx])
	A.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy, -dy])

U, S, V = numpy.linalg.svd(numpy.array(A, numpy.float32))
numpyHomography = V[-1, :].reshape(3, 3) / V[-1, -1]
# use a backward warping algorithm to warp the source
# to do so, we first create the inverse transform
# use bilinear interpolation for resampling
# in practice, cv2.warpPerspective can be used for this
# however, do not use this function for this exercise

numpyHomography = numpy.linalg.inv(numpyHomography)
numpyOutput = numpy.zeros(numpyInput.shape, numpy.float32)


# print(numpyInput.shape)
for intY in tqdm.tqdm(range(numpyInput.shape[0])):
	for intX in range(numpyInput.shape[1]):
		numpyDst = numpy.array([ intX, intY, 1.0 ], numpy.float32)

		numpySrc = numpy.matmul(numpyHomography, numpyDst.T)
		# print(numpySrc.shape)
		numpySrc = numpySrc / numpySrc[2]

		if numpySrc[0] < 0.0 or numpySrc[0] > numpyOutput.shape[1] - 1.0:
			continue

		elif numpySrc[1] < 0.0 or numpySrc[1] > numpyOutput.shape[0] - 1.0:
			continue

		# numpyOutput[intY, intX] = numpyInput[int(round(numpySrc[1])), int(round(numpySrc[0]))]
		alpha = (numpySrc[0] - int(numpy.floor(numpySrc[0])))
		beta = (numpySrc[1] - int(numpy.floor(numpySrc[1])))
		# print(numpySrc[0], numpySrc[1])
		numpyOutput[intY, intX] = (1 - alpha) * (1 - beta) * numpyInput[int(numpySrc[1]), int(numpySrc[0])] + (1 - alpha) * beta * numpyInput[int(numpySrc[1]) + 1, int(numpySrc[0])] + alpha * (1 - beta) * numpyInput[int(numpySrc[1]), int(numpySrc[0]) + 1] + alpha * beta * numpyInput[int(numpySrc[1]) + 1, int(numpySrc[0]) + 1]


cv2.imwrite(filename='./08-homography.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))