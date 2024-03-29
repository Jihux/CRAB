from torchvision import transforms

# Data Augmentation defaults
TRAIN_TRANSFORMS = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
