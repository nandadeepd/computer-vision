import numpy
import cv2

# this exercise references "Exposure Fusion" by Mertens et al.

numpyInputs = [
	cv2.imread(filename='./samples/fusion-1.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0,
	cv2.imread(filename='./samples/fusion-2.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0,
	cv2.imread(filename='./samples/fusion-3.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
]

# use the quality measures to extract a weight map for each image according to section 3.1
# set the weighting exponents to one, thus equaling the contrition of contrast, saturation, and exposedness
# make sure to add a small epsilon to each weight map to avoid divisions by zero in the subsequent step
# normalize the weight maps such that they sum up to one at each pixel as described in section 3.2
# store the three weight maps in the numpyWeights array which will be used below to perform the blending

# exp(−(i−0.5)^2/2 sigma^2)
# imageSaturation = lambda imageMat, channel : numpy.std(imageMat[:, :, channel])
contrastMeasure = lambda image : numpy.abs(cv2.Laplacian(src=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY), ddepth=-1))
saturationMeasure = lambda image : numpy.std(image, axis = 2)
sigma = 0.2
def calcExposedness(image):
	red, green, blue = image[:, :, 2], image[:, :, 1], image[:, :, 0]
	red = numpy.exp(-(red - 0.5) ** 2 / (2 * sigma ** 2))
	green = numpy.exp(-(green - 0.5) ** 2 / (2 * sigma ** 2))
	blue = numpy.exp(-(blue - 0.5) ** 2 / (2 * sigma ** 2))
	return red * green * blue

numpyWeights = []
# Collecting weights - abs of laplacian pyramids, saturation, well exposedness
def getWeightMap(w_c, w_s, w_e):

        weights = []
        sums = numpy.zeros(shape = (numpyInputs[0].shape[0], numpyInputs[0].shape[1]))
        for image in numpyInputs:
            contrast = contrastMeasure(image)
            # contrast = cv2.GaussianBlur(src=contrast, ksize=(201, 201), sigmaX=0.0, sigmaY=0.0, borderType=cv2.BORDER_DEFAULT)
            saturation = saturationMeasure(image)
            exposedness = calcExposedness(image)
            # applying the weighting exponents + the small eps value as directed
            weight = (contrast ** w_c) * (saturation ** w_s) * (exposedness ** w_e) + 0.0000001 # some small value
            weights.append(weight)
            sums = sums + weight
        # normalizing
        for index in range(len(numpyInputs)):
           	weights[index] = weights[index] / sums
        return weights

numpyWeights = getWeightMap(1, 1, 1)
# creating the laplacian and gaussian pyramids to perform multiband blending
# defining separate functions for this steps makes the code easier to read

def gaussian_pyramid(numpyInput, intLevels):
	numpyPyramid = [ numpyInput ]
	for intLevel in range(intLevels):
		numpyPyramid.append(cv2.pyrDown(numpyPyramid[-1]))
	# end
	return numpyPyramid
# end

def laplacian_pyramid(numpyInput, intLevels):
	numpyPyramid = [ numpyInput ]
	for intLevel in range(intLevels):
		numpyPyramid.append(cv2.pyrDown(numpyPyramid[-1]))
		numpyPyramid[-2] -= cv2.pyrUp(numpyPyramid[-1])
	# end
	return numpyPyramid
# end

numpyInputs = [ laplacian_pyramid(numpyInput, 6) for numpyInput in numpyInputs ]
numpyWeights = [ gaussian_pyramid(numpyWeight, 6) for numpyWeight in numpyWeights ]

# constructing a laplacian pyramid by using the weights from the gaussian pyramid
# eventually obtaining the fused result by recovering the output from the merged pyramid

numpyPyramid = []

for intLevel in range(len(numpyInputs[0])):
	numpyPyramid.append(sum([ numpyInputs[intInput][intLevel] * numpyWeights[intInput][intLevel][:, :, None] for intInput in range(len(numpyInputs)) ]))
# end

numpyOutput = numpyPyramid.pop(-1)

while len(numpyPyramid) > 0:
	numpyOutput = cv2.pyrUp(numpyOutput) + numpyPyramid.pop(-1)
# end

cv2.imwrite(filename='./13-fusion-1.png', img=(numpyWeights[0][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='./13-fusion-2.png', img=(numpyWeights[1][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='./13-fusion-3.png', img=(numpyWeights[2][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='./13-fusion-4.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))