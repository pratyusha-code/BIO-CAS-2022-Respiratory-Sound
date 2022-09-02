from data_loader_1_2 import *
from tqdm import tqdm
import numpy as np
# import spacy
import random
import sys
import pickle
import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from model_1_2 import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import collections

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

#Function to Save Checkpoint
def save_ckp(checkpoint, checkpoint_path):
	torch.save(checkpoint, checkpoint_path)

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, model_opt):
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['state_dict'])
	model_opt.load_state_dict(checkpoint['optimizer'])
	return model, model_opt, checkpoint['epoch']
	
			
def train_epoch(epoch, train_loader, model, criterion, model_opt):
	
	model.train()
	progress_bar = tqdm(enumerate(train_loader))
	total_loss = 0.0
	
	for step, (source_image, target_label) in progress_bar:
				
		try:
			model_opt.zero_grad()
			out = model.forward(source_image.to(device))
			loss = criterion(out, target_label.to(device))
			total_loss +=loss.item()
			loss.backward()
			model_opt.step()
			progress_bar.set_description("Epoch : {} Training Loss : {} Iteration : {}/{}".format(epoch+1, total_loss / (step + 1), step+1, len(train_loader))) 
			progress_bar.refresh()
		except:
			continue
		
	return total_loss/(step+1), model, model_opt            

def valid_epoch(epoch, valid_loader, model, criterion):
	
	model.eval()
	progress_bar = tqdm(enumerate(valid_loader))
	total_loss = 0.0
	total_tokens = 0
	with torch.no_grad():
		for step, (source_image, target_label) in progress_bar:
					
			# source_image = source_image.squeeze(1)
			out = model.forward(source_image.to(device))
			loss = criterion(out, target_label.to(device))
			total_loss +=loss
			progress_bar.set_description("Epoch : {} Validation Loss : {} Iteration : {}/{}".format(epoch+1, total_loss / (step + 1), step+1, len(valid_loader))) 
			progress_bar.refresh()  
		
	return total_loss/(step+1)

def test_epoch(epoch, test_loader, model, criterion):
	
	os.makedirs("confusion_matrix_1_2", exist_ok="True")
	model.eval()
	progress_bar = tqdm(enumerate(test_loader))
	total_loss = 0.0
	total_tokens = 0
	target_list = []
	predicted_list = []
	with torch.no_grad():

		for step, (source_image, target_label) in progress_bar:
			# source_image = source_image.squeeze(1)
			out = model.forward(source_image.to(device))
			pred = list(torch.argmax(out, dim=1).cpu().numpy())
			target = list(target_label.cpu().numpy())
			predicted_list.extend(pred)
			target_list.extend(target)
	print(accuracy_score(predicted_list, target_list)*100)
	print(confusion_matrix(target_list, predicted_list))

	df_cm = pd.DataFrame(np.array(confusion_matrix(target_list, predicted_list)), index = [i for i in ['Coarse Crackle','Fine Crackle','Normal','Rhonchi','Stridor','Wheeze','Wheeze+Crackle']],
                  columns = [i for i in ['Coarse Crackle','Fine Crackle','Normal','Rhonchi','Stridor','Wheeze','Wheeze+Crackle']])

	plt.figure(figsize = (10,10))
	sn.heatmap(df_cm, annot=True, cmap="Greens")
	plt.title("Epoch : {}    Accuracy : {:.2f}%".format(epoch+1, accuracy_score(predicted_list, target_list)*100))
	plt.savefig(os.path.join("confusion_matrix_1_2", str(epoch)+".png"))

	return accuracy_score(predicted_list, target_list)*100

def init_weights(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)

def training_testing(train_loader, test_loader, model, criterion, model_opt, resume):
	
	epoch = 0
	checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/temp_task_1_2/"
	os.makedirs(checkpoint_dir, exist_ok=True)

	checkpoint_path = checkpoint_dir + "checkpoint_1_2.pt"
	if resume:
		model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
		resume = False
		print("Resuming Training from Epoch Number : ", epoch)
	checkpoint_duplicate_dir = "/scratch/seshadri_c/temp_task_1_2/"
	os.makedirs(checkpoint_duplicate_dir, exist_ok=True)

	checkpoint_duplicate_path = checkpoint_duplicate_dir + "checkpoint_" + str(epoch + 1) + ".pt"
	while(1):

		checkpoint_path = checkpoint_dir + "checkpoint_latest.pt"
		checkpoint_duplicate_path = checkpoint_duplicate_dir + "checkpoint_" + str(epoch + 1) + ".pt"

		print("\n\nTraining : ")
		train_loss, model, model_opt = train_epoch(epoch, train_loader, model, criterion, model_opt)

		# Creating the Checkpoint
		checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': model_opt.state_dict()}

		# Saving the Checkpoint
		save_ckp(checkpoint, checkpoint_path)
		print("Saved Successfully")
		save_ckp(checkpoint, checkpoint_duplicate_path)
		print("Duplicate Checkpoint Saved Successfully")

		#Loading the Checkpoint
		# model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
		# print("Loaded Successfully")
		
		print("Testing : ")
		valid_loss = valid_epoch(epoch, test_loader, model, criterion)
		test_accuracy = test_epoch(epoch, test_loader, model, criterion)

		print("Epoch No {} completed.".format(epoch + 1))
		print("Train Loss : {} \t Valid Loss : {} \t Accuracy : {}".format(train_loss, valid_loss, test_accuracy))
		if(epoch==0):
			with open('train_valid_loss.txt', 'w') as f:
				f.write("Epoch : {} \t Train Loss : {} \t Valid Loss : {} \t Test Accuracy : {}\n".format(epoch, train_loss, valid_loss, test_accuracy))
		else:
			with open('train_valid_loss.txt', 'a') as f:
				f.write("Epoch : {} \t Train Loss : {} \t Valid Loss : {} \t Test Accuracy : {}\n".format(epoch, train_loss, valid_loss, test_accuracy))
		
		epoch+=1

def make_dataset_equivalent(x, y):

	data_dict = {}
	for i in range(len(y)):
		if y[i] not in data_dict.keys():
			data_dict[y[i]] = [x[i]]
		else :
			data_dict[y[i]].append(x[i])

	
	max_val = max([len(data_dict[k]) for k in data_dict.keys()])

	print("Number of Samples before Sampling : ")

	for k in data_dict.keys():
		print(k, len(data_dict[k]))
	print(max_val)

	for k in data_dict.keys():

		list_data = data_dict[k]
		new_list_data = list(np.random.choice(list_data, size=max_val))
		data_dict[k] = new_list_data

	print("Number of Samples After Sampling : ")

	for k in data_dict.keys():
		print(k, len(data_dict[k]))
	print(max_val)


	sampled_data = []
	for k in data_dict.keys():
		for val in data_dict[k]:
			sampled_data.append((val, k))

	random.shuffle(sampled_data)
	x = []
	y = []

	for a, b in sampled_data:
		x.append(a)
		y.append(b)

	return x, y

def main():
	
	task = 1 
	if(task==1):
		data_folder = "data/event_level"
		file_list = glob.glob(data_folder+"/*/*/*.png")

	if(task==2):
		data_folder = "data/record_level"
		file_list = glob.glob(data_folder+"/*/*/*.png")

	x = []
	y = []
	for f in file_list:
		x.append(f)
		y.append(f.split('/')[-2].strip())

	labels = list(set(y))
	labels.sort()

	id2label={i: c for i, c in enumerate(labels)},
	label2id={c: i for i, c in enumerate(labels)}
	print(label2id)

	y_new = []
	for t in y:
		y_new.append(label2id[t])
	x, y = make_dataset_equivalent(x, y_new)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)


	print(len(x_train), len(x_test), len(y_train), len(y_test))

	model = Classifier()
	model = model.to(device)

	# model = nn.DataParallel(model)	
	
	model_opt = torch.optim.Adam(model.parameters(),lr=0.0001)
	 
	criterion = nn.CrossEntropyLoss()
	criterion.cuda()
	
	train_loader = load_data(x_train, y_train, batch_size=128, num_workers=10, shuffle=True)
	test_loader = load_data(x_test, y_test, batch_size=128, num_workers=10, shuffle=False)
	resume = False
	
	training_testing(train_loader, test_loader, model, criterion, model_opt, resume)

main()