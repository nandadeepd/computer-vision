import numpy
import cv2
import os
import zipfile
import matplotlib.pyplot

# this exercise references "Photographic Tone Reproduction for Digital Images" by Reinhard et al.

numpyRadiance = cv2.imread(filename='./samples/ahwahnee.hdr', flags=-1)
# perform tone mapping according to the photographic luminance mapping
# first extracting the intensity from the color channels
# note that the eps is to avoid divisions by zero and log of zero
numpyIntensity = cv2.cvtColor(src=numpyRadiance, code=cv2.COLOR_BGR2GRAY) + 0.0000001  # luminance

# start off by approximating the key of numpyIntensity according to equation 1
# then normalize numpyIntensity using a = 0.18 according to equation 2
# afterwards, apply the non-linear tone mapping prescribed by equation 3
# finally obtain numpyOutput using the ad-hoc formula with s = 0.6 from the slides
delta = 0.0001  # some small value 
N = numpyIntensity.size
summation = 0
numpyRadianceIn = numpy.copy(numpyRadiance)

for x in range(numpyIntensity.shape[0]):
	for y in range(numpyIntensity.shape[1]):
		summation += (numpy.log((delta + numpyIntensity[x,y])))
Lbar = numpy.exp(summation/N) ## Equation 1

for x in range(numpyIntensity.shape[0]):
	for y in range(numpyIntensity.shape[1]):
		numpyIntensity[x, y] = (0.18/Lbar) * numpyIntensity[x, y]  ## Equation 2

scaled = numpy.empty_like(numpyIntensity)
scaled = numpyIntensity[:, :] / 1 + numpyIntensity[:,:]  ## Equation 3
numpyOutput = numpy.empty(shape=(numpyIntensity.shape[0], numpyIntensity.shape[1], 3))
for color in range(3):
	numpyOutput[:,:,color] = numpy.power((numpyRadianceIn[:, :, color] / numpyIntensity), 0.6) * scaled

# Cout = numpy.power((Cin/Lin), 0.6) * Lout

cv2.imwrite(filename='./12-tonemapping.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))