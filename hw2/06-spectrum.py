import numpy
import cv2
import matplotlib.pyplot

numpyImage = cv2.imread(filename='./samples/lenna.png', flags=cv2.IMREAD_GRAYSCALE).astype(numpy.float32) / 255.0

# creating a relief kernel that you are subsequently asked to apply in the frequency space
numpyKernel = numpy.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], numpy.float32)

# pad numpyKernel such that its resolution is equal to the resolution of numpyImage
# numpyKernel = numpy.pad(numpyKernel, (255,254), 'constant') #hit and trial

# roll it one pixel to the left and one pixel to the top in order to adjust the phase
numpyKernel = numpy.roll(numpy.roll(numpy.pad(numpyKernel, [(0, numpyImage.shape[0] - 3), (0, numpyImage.shape[1] - 3)], 'constant'), -7, 0), -7, 1)

# convert numpyKernel into the frequency space by using the fourier transform
numpyKernel = numpy.fft.fft2(numpyKernel)

# similarly, convert numpyImage into the frequency space, no need to pad or roll it though
numpyImage = numpy.fft.fft2(numpyImage)
matplotlib.pyplot.imshow(numpy.log(numpy.abs(numpy.fft.fftshift(numpyImage)) + 1.0))
matplotlib.pyplot.show()

# perform the convolution in the frequency space by using the hadamard product
numpyImage = numpyImage * numpyKernel

# convert the result back to the spacial domain by using the inverse fourier transform
numpyImage = numpy.fft.ifft2(numpyImage)
matplotlib.pyplot.imshow(numpy.log(numpy.abs(numpy.fft.fftshift(numpyImage)) + 1.0))
matplotlib.pyplot.show()

# the result / numpyImage should look as if the convolution was done in the spatial domain
# plotting the frequency spectrum of the gaussian kernel while making sure that we only plot the real part
# the logarithmic scaling improves the visualizaation, the constant bias avoids log(0-1) which is undefined / negative
# the value range for the spectrum is outside of [0, 1] and a color mapping is applied before saving it

numpySpectrum = numpy.log(numpy.abs(numpy.fft.fftshift(numpyKernel)) + 1.0)
numpySpectrum = numpySpectrum / numpySpectrum.max()
numpyImage = numpyImage.real


cv2.imwrite(filename='./06-spectrum-1.png', img=cv2.applyColorMap(src=(numpySpectrum * 255.0).clip(0.0, 255.0).astype(numpy.uint8), colormap=cv2.COLORMAP_WINTER))
cv2.imwrite(filename='./06-spectrum-2.png', img=(numpyImage * 255.0).clip(0.0, 255.0).astype(numpy.uint8))