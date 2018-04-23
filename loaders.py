import torch
from torchvision import datasets, transforms, models

def load(data_dir):
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'

	train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

	test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

	train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
	valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

	num_train_images = len(train_loader.dataset.imgs)
	num_valid_images = len(valid_loader.dataset.imgs)

	return train_loader, valid_loader, num_train_images, num_valid_images
