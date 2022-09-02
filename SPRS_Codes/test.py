import os
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange

def test_model(model_name, model, save_path, test_dataloader, t, optimizer, criterion, device):
	epoch_list = []
	for i in os.listdir(save_path):
		epoch_list.append(int(i.split('.')[0].split('_')[-1]))
	print(f"Testing {model_name} on {device} after {max(epoch_list)} epochs")
	model.load_state_dict(torch.load(f'{save_path}model_{t}_epoch_{max(epoch_list)}.pt'))

	error = 0

	for l, data in tqdm(enumerate(test_dataloader, 0)):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = model(inputs.to(device))
		output = np.zeros(outputs.cpu().shape)
		for i in range(len(outputs)):
			for j in range(len(outputs[0])):
				k = outputs[i][j].cpu()
				output[i][j] = k
		if labels.cpu()!=list(output[i]).index(max(output[0])):
			error = error + 1
	accuracy = 100*(1-error/len(test_dataloader.dataset))
	print(f'Accuracy for {model_name} is {accuracy}')
	return


def test(model_name, model, save_path, s_db, t,optimizer, criterion, device):
	epoch_list = []
	# for i in os.listdir(save_path):
	# 	epoch_list.append(int(i.split('.')[0].split('_')[-1]))
	# print(f"Testing {model_name} on {device} after {max(epoch_list)} epochs")
	# print(save_path)
	model.load_state_dict(torch.load(save_path))

	error = 0
	inputs = s_db
	optimizer.zero_grad()
	inputs = torch.Tensor(inputs)
	inputs = torch.unsqueeze(inputs,0)
	inputs = rearrange(inputs, 'b h w c -> b c h w')
	print(inputs.shape)
	output = model(inputs.to(device))
	# output = np.zeros(outputs.cpu().shape)
	# for i in range(len(outputs)):
		# for j in range(len(outputs[0])):
			# k = outputs[i][j].cpu()
			# output[i][j] = k
	print(output.shape,"Hello", np.argmax(output[0].cpu().detach().numpy()))
	return np.argmax(output[0].cpu().detach().numpy())
