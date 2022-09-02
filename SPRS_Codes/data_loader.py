import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def data_loader(batch_size, data_dir):
	transform = transforms.Compose([transforms.Resize(255),
									transforms.CenterCrop(224),
									transforms.ToTensor()])
	dataset = datasets.ImageFolder(data_dir, transform=transform)
	train_dataset, test_dataset = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
	print("Data loaded successfully!")
	return train_dataloader,test_dataloader

def test_data_loader(batch_size, data_dir):
	transform = transforms.Compose([transforms.Resize(255),
									transforms.CenterCrop(224),
									transforms.ToTensor()])
	dataset = datasets.ImageFolder(data_dir, transform=transform)
	test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
	print("Data loaded successfully!")
	return test_dataloader
