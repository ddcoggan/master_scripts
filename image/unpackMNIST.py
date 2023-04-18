import os
import cv2
import numpy as np

train_image = 'train-images-idx3-ubyte'
train_label = 'train-labels-idx1-ubyte'
test_image = 't10k-images-idx3-ubyte'
test_label = 't10k-labels-idx1-ubyte'

for f in [train_image, train_label, test_image, test_label]:
	os.system('wget --no-check-certificate http://yann.lecun.com/exdb/mnist/%s.gz' % (f,))

for f in [train_image, train_label, test_image, test_label]:
	os.system('gunzip %s.gz' % (f,))

for image_f, label_f in [(train_image, train_label), (test_image, test_label)]:
	with open(image_f, 'rb') as f:
		images = f.read()
	with open(label_f, 'rb') as f:
		labels = f.read()
	
	images = [d for d in images[16:]]
	images = np.array(images, dtype=np.uint8)
	images = images.reshape((-1,28,28))
	labels = ['%d' % (l) for l in labels[8:]]

	if len(labels) == 60000:
		trainVal = 'train'
	elif len(labels) == 10000:
		trainVal = 'val'

	for i, image in enumerate(images):

		label = labels[i]
		outDir = os.path.join(trainVal, label)
		os.makedirs(outDir, exist_ok=True)
		cv2.imwrite(os.path.join(outDir, '%05d.png' % (i,)), image)


