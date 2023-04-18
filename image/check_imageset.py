# used to find image giving the corrupt exif data warning when using DNNs
import glob
from PIL import Image
imageDir = '/home/dave/Datasets/imagenet1000'
images = sorted(glob.glob(f'{imageDir}/*/*/*', recursive=True))
print(f'{len(images)} images found')
sizes = []
for i, image in enumerate(images):
    im = Image.open(image)
# image is n04152593_17064
