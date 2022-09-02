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
import torch.nn as nnt
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, FrequencyMasking
from tqdm import tqdm

from utils import cuda_gpu, plot_spectrogram_train_events, plot_spectrogram_train_records


host = 'ada'
train_on_gpu, multi_gpu, device = cuda_gpu(host)


#############Dataprep####################################

json_dir = "/scratch/adithyasunil/SPRSound/train_json"
os.chdir(json_dir)
json_files = [file for file in glob.glob("*.json")]
audio_dir = "/scratch/adithyasunil/SPRSound/train_wav"
os.chdir(audio_dir)
audio_files = [file for file in glob.glob("*.wav")]
print(len(json_files))
print(len(audio_files))
json_files.sort()
audio_files.sort()


##############Events#########################

print("Generating event data spectrograms")

event_data_dir = "/scratch/adithyasunil/event"
os.makedirs(event_data_dir, exist_ok=True)

for i in tqdm(range(len(json_files))):
	with open(os.path.join(json_dir, json_files[i])) as f:
		json_f = json.load(f)['event_annotation']
		# print(json_files[i])
		for j in range(len(json_f)):
			s = int(json_f[j]['start'])
			e = int(json_f[j]['end'])
			t = json_f[j]['type']

			if e-s==0:
				continue

			os.makedirs(os.path.join(event_data_dir, audio_files[i][:-4]), exist_ok=True)
			y, sr = torchaudio.load(os.path.join(audio_dir, audio_files[i]))

			s_n = s
			e_n = e

			for k in range(3):
				if k!=0:
					s_n = s + random.randint(-10,10)
					e_n = e + random.randint(-10,10)

				y_segment = y[0][s_n:e_n]

				try:
					S = MelSpectrogram(sample_rate=sr, n_fft=128, win_length=32, hop_length=32, pad=1, n_mels=32, mel_scale="htk", center=True, pad_mode="reflect", norm="slaney")(y_segment)
					S_dB = AmplitudeToDB(stype='power')(S)
					S_dB_f = S_dB
					
					# print("Shape of spectrogram: {}".format(S_dB.size()))

					for l in range(3):
						if l==0:
							# print("Shape of spectrogram: {}".format(S_dB.size()))
							plot_spectrogram_train_events(S_dB, t, i, j, k, l, event_data_dir)
						else:
							S_dB_f = FrequencyMasking(freq_mask_param=2)(S_dB)
							plot_spectrogram_train_events(S_dB_f, t, i, j, k, l, event_data_dir)
							# print(i,j,k,l)
				except:
					# print("skipped")
					continue
	# print("Completed : {}/{}".format(i+1, len(json_files)))
		f.close()

os.chdir(event_data_dir)
os.system("rm -rf *_*_*")

#############Records####################################

print("Generating record data spectrograms")

record_data_dir = "/scratch/adithyasunil/record"
os.makedirs(record_data_dir, exist_ok=True)

for i in tqdm(range(len(json_files))):

	with open(os.path.join(json_dir, json_files[i])) as f:
		rec_t = json.load(f)['record_annotation']

		y, sr = torchaudio.load(os.path.join(audio_dir, audio_files[i]))

		# try:
		S = MelSpectrogram(sample_rate=sr, n_fft=2048, win_length=128, hop_length=32, pad=1, n_mels=128, mel_scale="htk", pad_mode="reflect", norm="slaney")(y)
		S_dB = AmplitudeToDB(stype='power')(S)
		S_dB_f = S_dB

		for l in range(3):
			if l==0:
				plot_spectrogram_train_records(S_dB, rec_t, i, l, record_data_dir)
			else:
				S_dB_f = FrequencyMasking(freq_mask_param=16)(S_dB)
				plot_spectrogram_train_records(S_dB_f, rec_t, i, l, record_data_dir)
		# except:
			# continue
	f.close()

os.chdir(record_data_dir)
os.system("rm -rf *_*_*")
