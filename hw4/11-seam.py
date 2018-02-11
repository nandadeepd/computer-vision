import numpy
import cv2

# this exercise references "Seam Carving for Content-Aware Image Resizing" by Avidan and Shamir

numpyInput = cv2.imread(filename='./samples/seam.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# implement content-aware image resizing to reduce the width of the image by one-hundred pixels
# using a heuristic energy function to extract an energy map

numpyEnergy = numpy.abs(cv2.Sobel(src=cv2.cvtColor(src=numpyInput, code=cv2.COLOR_BGR2GRAY), ddepth=-1, dx=1, dy=0, ksize=3, scale=1, delta=0.0, borderType=cv2.BORDER_DEFAULT)) \
			+ numpy.abs(cv2.Sobel(src=cv2.cvtColor(src=numpyInput, code=cv2.COLOR_BGR2GRAY), ddepth=-1, dx=0, dy=1, ksize=3, scale=1, delta=0.0, borderType=cv2.BORDER_DEFAULT))

# find and remove one-hundred vertical seams, can potentially be slow
for _ in range(100):
	intSeam = []

	# construct the cumulative energy map using the dynamic programming approach
	# initialize the cumulative energy map by making a copy of the energy map
	# when iterating over the rows, ignore M(y-1, ...) that are out of bounds
	energyMap = numpy.copy(numpyEnergy)
	# several seams can have the same energy, use the following for consistency
	# start at the leftmost M(height-1, x) with the lowest cumulative energy
	# should M(y-1, x) be equal to M(y-1, x-1) or M(y-1, x+1) then use (y-1, x)
	# similarly should M(y-1, x-1) be equal to M(y-1, x+1) then use (y-1, x-1)

	# the intSeam array should be a list of integers representing the seam
	# a seam from the top left to the bottom right: intSeam = [0, 1, 2, 3, 4, ...]
	# a seam that is just the first column: intSeam = [0, 0, 0, 0, 0, 0 , ...]

	height, width = numpyEnergy.shape
	energyMap = numpy.copy(numpyEnergy)
	for row in range(1, energyMap.shape[0]):
		for column in range(energyMap.shape[1]):
			if column == 0:
				energyMap[row, column] += min(energyMap[row - 1, column], energyMap[row - 1, column + 1])
			elif column == (energyMap.shape[1] - 1):
				energyMap[row, column] += min(energyMap[row - 1, column - 1], energyMap[row - 1, column])
			else:
				# finding min of all three values since we have them. NOT edge cases
				energyMap[row, column] += min(energyMap[row - 1, column - 1], energyMap[row - 1, column], energyMap[row - 1, column + 1])

	startingPoint = numpy.argmin(energyMap[height - 1, :])
	intSeam.append(startingPoint)

	for i in range(height - 2, -1, -1):
		# checking height border cases
		if startingPoint == 0:
			minval = numpy.argmin(numpy.array([energyMap[i,startingPoint],energyMap[i,startingPoint + 1]]))
			startingPoint += (minval)
			intSeam.append(startingPoint)
		# checking width border cases
		elif startingPoint == width-1:
			minval = numpy.argmin(numpy.array([energyMap[i,startingPoint-1],energyMap[i,startingPoint]]))
			startingPoint += (minval - 1)
			intSeam.append(startingPoint)
		else:
			minval = numpy.argmin(numpy.array([energyMap[i,startingPoint-1],energyMap[i,startingPoint],energyMap[i,startingPoint + 1]]))
			startingPoint += (minval - 1)
			intSeam.append(startingPoint)

	# flipping to match shape
	intSeam = intSeam[ : : -1]
	
	# some sanity checks, such that the length of the seam is equal to the height of the image
	# furthermore iterating over the seam and making sure that it is a connected sequence

	assert(len(intSeam) == numpyInput.shape[0])
	for intY in range(1, len(intSeam)):
		assert(intSeam[intY] - intSeam[intY - 1] in [-1, 0, 1])
	# end

	# change the following condition to true if you want to visualize the seams that are being removed
	# note that this will not work if you are connected to the linux lab via ssh but no x forwarding

	if True:
		for intY in range(len(intSeam)):
			numpyInput[intY, intSeam[intY], :] = numpy.array([ 0.0, 0.0, 1.0 ], numpy.float32)
		# end

		cv2.imshow(winname='numpyInput', mat=numpyInput)
		cv2.waitKey(delay=10)
	# end

	# removing the identified seam by iterating over each row and shifting them accordingly
	# after the shifting in each row, the image and the energy map are cropped by one pixel on the right

	for intY in range(len(intSeam)):
		numpyInput[intY, intSeam[intY]:-1, :] = numpyInput[intY, (intSeam[intY] + 1 ):, :]
		numpyEnergy[intY, intSeam[intY]:-1] = numpyEnergy[intY, (intSeam[intY] + 1):]
	# end

	numpyInput = numpyInput[:, :-1, :]
	numpyEnergy = numpyEnergy[:, :-1]
# end
cv2.imwrite(filename='./11-seam.png', img=(numpyInput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))