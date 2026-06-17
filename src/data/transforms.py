from torchvision import transforms


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])