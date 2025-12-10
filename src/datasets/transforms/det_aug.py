from torchvision import transforms
from src.utils.common import get_normalization


def build_detection_transforms(img_size: int = 640):
    mean, std = get_normalization("detection")
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf, tf
