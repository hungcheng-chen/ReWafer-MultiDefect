import torchvision.transforms as T


def BASIC_TRANSFORMS():
    """
    Basic image transformations: resize, convert to tensor, and normalize.
    """
    img_size: list = [224, 224]
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.Resize(img_size),  # Resize to 224x224
            T.ToTensor(),  # Convert to tensor
            T.Normalize(mean=mean, std=std),  # Normalize with ImageNet stats
        ]
    )


def AUGMENTATION_TRANSFORMS():
    """
    Augmentation transformations: resize, random rotations/flips, convert to tensor, and normalize.
    """
    img_size: list = [224, 224]
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.Resize(img_size),  # Resize to 224x224
            T.RandomRotation((-45, 45), fill=(255, 255, 255)),  # Random rotation
            T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            T.RandomVerticalFlip(p=0.5),  # Random vertical flip
            T.ToTensor(),  # Convert to tensor
            T.Normalize(mean=mean, std=std),  # Normalize with ImageNet stats
        ]
    )
