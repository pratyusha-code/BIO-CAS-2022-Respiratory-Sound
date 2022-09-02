import torch
from tqdm import tqdm

def train_model(model_name, model, save_path, train_dataloader, epochs, optimizer, criterion, device, test):
	if test == 1:
		print("Test mode: Skipping training!")
		return model
	print(f"Training {model_name} for {epochs} epochs on {device}")
	for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
	#     print("Epoch ",epoch+1)
		running_loss = 0.0
		for i, data in enumerate(tqdm(train_dataloader), 0):
			
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			optimizer.zero_grad()
			outputs = model(inputs.to(device))
			loss = criterion(outputs, labels.to(device))
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
	#         if epoch%10==0:
	#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
			running_loss = 0.0
		if (epoch+1)%5==0:
			chkp_save_path = save_path + "model_event_epoch_" + str(epoch+1) + ".pt" 
			torch.save(model.state_dict(), chkp_save_path)
		if epochs == 1:
			chkp_save_path = save_path + "model_event_epoch_" + str(epoch+1) + ".pt" 
			torch.save(model.state_dict(), chkp_save_path)
	print(f'Finished Training {model_name} for {epochs} epochs on {device}')
	return model