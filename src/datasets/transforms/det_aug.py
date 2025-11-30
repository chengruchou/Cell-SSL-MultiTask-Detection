import torchvision.transforms.functional as F
import random
from PIL import Image

class DetAug:
    def __init__(self, size=640):
        self.size = size

    def __call__(self, img, boxes):
        # Random flip
        if random.random() < 0.5:
            img = F.hflip(img)
            boxes[:,0] = 1 - boxes[:,0]

        # Resize
        img = img.resize((self.size, self.size), Image.BILINEAR)

        return F.to_tensor(img), boxes
