import torchvision.transforms as T


def BASIC_TRANSFORMS():
    img_size: list = [224, 224]

    # ImageNet mean and std
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def AUGMENTATION_TRANSFORMS():
    img_size: list = [224, 224]

    # ImageNet mean and std
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize(img_size),
            # T.RandomResizedCrop(224),
            # T.RandomRotation(15),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.RandomGrayscale(p=0.2),  # 添加随机灰度转换
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )
