"""
Common utilities shared across tasks.
"""


def get_normalization(task: str):
    """
    Return (mean, std) for a given task.
    classification/detection/segmentation -> ImageNet stats
    ssl -> MAE-style normalization
    """
    task = task.lower()
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    mae_mean = (0.5, 0.5, 0.5)
    mae_std = (0.5, 0.5, 0.5)

    if task in ["classification", "detection", "segmentation"]:
        return imagenet_mean, imagenet_std
    if task == "ssl":
        return mae_mean, mae_std
    raise ValueError(f"Unknown task for normalization: {task}")
