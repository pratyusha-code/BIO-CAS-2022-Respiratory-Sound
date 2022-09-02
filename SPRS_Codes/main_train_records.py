import os
import glob
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import cuda
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, FrequencyMasking
from tqdm import tqdm

from utils import cuda_gpu, plot_spectrogram_train_records, imshow, plot_dataset, model_summary
from data_loader import data_loader
from model_script import get_pretrained_model
from train import train_model
from test import test_model


host = 'ada'
train_on_gpu, multi_gpu, device = cuda_gpu(host)


#############Dataprep####################################

# json_dir = "/scratch/adithyasunil/SPRSound/train_json"
# os.chdir(json_dir)
# json_files = [file for file in glob.glob("*.json")]
# audio_dir = "/scratch/adithyasunil/SPRSound/train_wav"
# os.chdir(audio_dir)
# audio_files = [file for file in glob.glob("*.wav")]
# print(len(json_files))
# print(len(audio_files))
# json_files.sort()
# audio_files.sort()


##############Recordss#########################

print("Generating record data spectrograms")

record_data_dir = "/scratch/adithyasunil/scratch/adithyasunil/record"
# os.makedirs(record_data_dir, exist_ok=True)

# for i in tqdm(range(len(json_files))):
# 	with open(os.path.join(json_dir, json_files[i])) as f:
# 		rec_t = json.load(f)['record_annotation']

# 		y, sr = torchaudio.load(os.path.join(audio_dir, audio_files[i]))

# 		# Resampling to 44100 Hz
# 		# y_rs = Resample(sr, targ_sr, dtype=y.dtype)(y)

# 		try:
# 		# Generating spectrogram
# 			S = MelSpectrogram(sample_rate=sr, n_fft=2048, win_length=128, 
# 							hop_length=32, pad=1, n_mels=128, mel_scale="htk", 
# 							pad_mode="reflect", norm="slaney")(y)
# 			S_dB = AmplitudeToDB(stype='power')(S)
# 			S_dB_f = S_dB

# 			for l in range(3):
# 				if l==0:
# 					print("Shape of spectrogram: {}".format(S_dB.size()))
# 					plot_spectrogram(S_dB, rec_t, i, l, record_data_dir)
# 					# print("Done0")
# 				else:
# 					S_dB_f = FrequencyMasking(freq_mask_param=16)(S_dB)
# 					plot_spectrogram(S_dB_f, rec_t, i, l, record_data_dir)
# 					# print("Done1")  
# 		except:
# 			continue

# 	f.close()

# os.chdir(record_data_dir)
# os.system("rm -rf *_*_*")

#############Training####################################

categories = os.listdir(record_data_dir)
n_classes = len(categories)

batch_size = 128
train_dataloader,test_dataloader = data_loader(batch_size, record_data_dir)

# plot_dataset(train_dataloader, 8, 4)

model_list = ['vgg16', 'resnet50', 'mobilenet']

# Flag to be turned on for testing script changes
test = 0

# for model_name in model_list:
model_name = 'vgg16'
# model_name = 'resnet50'
# model_name = 'mobilenet'

model = get_pretrained_model(model_name, n_classes, multi_gpu, train_on_gpu, device)

input_size = (3, 224, 224)
# model_summary(model, multi_gpu, input_size, batch_size, device)
criterion = nn.CrossEntropyLoss()
criterion.cuda()
if model_name == 'resnet50':
	if multi_gpu:
		optimizer = optim.Adam(model.module.fc.parameters(), lr=0.003)
	elif train_on_gpu:
		optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
else:
	if multi_gpu:
		optimizer = optim.Adam(model.module.classifier.parameters(), lr=0.003)
	elif train_on_gpu:
		optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

save_path = f"/scratch/adithyasunil/model_checkpoints_SPRS/record_classification/{model_name}/"

epochs = 100

model = train_model(model_name, model, save_path, train_dataloader, epochs, optimizer, criterion, device, test)

test_model(model_name, model, save_path, test_dataloader, optimizer, criterion, device)

# # for model_name in model_list:
# model_name = 'vgg16'
# # model_name = 'resnet50'
# # model_name = 'mobilenet'

# model = get_pretrained_model(model_name, n_classes, multi_gpu, train_on_gpu, device)
# criterion = nn.CrossEntropyLoss()
# criterion.cuda()
# if model_name == 'resnet50':
# 	if multi_gpu:
# 		optimizer = optim.Adam(model.module.fc.parameters(), lr=0.003)
# 	elif train_on_gpu:
# 		optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
# else:
# 	if multi_gpu:
# 		optimizer = optim.Adam(model.module.classifier.parameters(), lr=0.003)
# 	elif train_on_gpu:
# 		optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

# save_path = f"/scratch/adithyasunil/model_checkpoints/sprs/event_classification/{model_name}/"

# test_model(model_name, model, save_path, test_dataloader, optimizer, criterion, device)
