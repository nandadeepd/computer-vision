import numpy
import cv2

# this exercise references "Color Transfer between Images" by Reinhard et al.

numpyFrom = cv2.imread(filename='./samples/transfer-from.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
numpyTo = cv2.imread(filename='./samples/transfer-to.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

numpyFrom = cv2.cvtColor(src=numpyFrom, code=cv2.COLOR_BGR2Lab)
numpyTo = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_BGR2Lab)

Lfrom, Afrom, Bfrom = numpyFrom[:,:,2], numpyFrom[:,:,1], numpyFrom[:,:,0]
Lto, Ato, Bto = numpyTo[:,:,2], numpyTo[:,:,1], numpyTo[:,:,0]
# match the color statistics of numpyTo to those of numpyFrom
# calculate the per-channel mean of the data points / pixels of numpyTo, and subtract these from numpyTo according to equation 10

LtoMean, AtoMean, BtoMean = Lto.mean(), Ato.mean(), Bto.mean()
Lto -= LtoMean
Ato -= AtoMean
Bto -= BtoMean

# calculate the per-channel std of the data points / pixels of numpyTo and numpyFrom, and scale numpyTo according to equation 11
lfromStd, afromStd, bfromStd = Lfrom.std(), Afrom.std(), Bfrom.std()
LtoStd, AtoStd, BtoStd = Lto.std(), Ato.std(), Bto.std()

Lto = (lfromStd/LtoStd) * Lto
Ato = (afromStd/AtoStd) * Ato
Bto = (bfromStd/BtoStd) * Bto

# calculate the per-channel mean of the data points / pixels of numpyFrom, and add these to numpyTo according to the description after equation 11
LfromMean, AfromMean, BfromMean = Lfrom.mean(), Afrom.mean(), Bfrom.mean()
Lto += LfromMean
Ato += AfromMean
Bto += BfromMean

numpyTo = numpy.stack([Bto, Ato, Lto], axis=2)
numpyTo[:, :, 0] = numpyTo[:, :, 0].clip(0.0, 100.0)
numpyTo[:, :, 1] = numpyTo[:, :, 1].clip(-127.0, 127.0)
numpyTo[:, :, 2] = numpyTo[:, :, 2].clip(-127.0, 127.0)

numpyOutput = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_Lab2BGR)

cv2.imwrite(filename='./02-colortransfer.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))