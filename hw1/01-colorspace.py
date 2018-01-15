import numpy
import cv2

# this exercise references "Color Transfer between Images" by Reinhard et al.

numpyInput = cv2.imread(filename='./samples/fruits.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# convert numpyInput to the LMS color space and store it in numpyOutput according to equation 4


multiplier = numpy.array([[0.3811, 0.5783, 0.0402], 
							[0.1967, 0.7244, 0.0782], 
							[0.0241, 0.1288, 0.8444]])

# print(multiplier[0][0]) -- debugging
# the two ways i see for doing this (there are others as well though) are as follows
# either iterate over each pixel, performing the matrix-vector multiplication one by one and storing the result in a pre-allocated numpyOutput
# or split numpyInput into its three channels, linearly combining them to obtain the three converted color channels, before using numpy.stack to merge them

# using the second suggested method. 
L = numpyInput[:,:,2] * multiplier[0][0] + numpyInput[:,:,1] * multiplier[0][1] + numpyInput[:,:,0] * multiplier[0][2]
M = numpyInput[:,:,2] * multiplier[1][0] + numpyInput[:,:,1] * multiplier[1][1] + numpyInput[:,:,0] * multiplier[1][2]
S = numpyInput[:,:,2] * multiplier[2][0] + numpyInput[:,:,1] * multiplier[2][1] + numpyInput[:,:,0] * multiplier[2][2]

numpyOutput = numpy.stack([L, M, S], axis=2)

cv2.imshow('ColorSpaceConverted', numpyOutput)
cv2.waitKey(0)

# keep in mind that that opencv arranges the color channels typically in the order of blue, green, red
cv2.imwrite(filename='./01-colorspace.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))