import cv2, numpy, os, re

desired_size = (384, 269)
images = [image for image in os.listdir("/Users/Nandadeep/Desktop/imgs") if re.search(r".png", image)]
resized_images = [cv2.resize(cv2.imread(image), desired_size) for image in images]

for _ in range(len(resized_images)):
	cv2.imwrite(str(_) + '.png', resized_images[_])
