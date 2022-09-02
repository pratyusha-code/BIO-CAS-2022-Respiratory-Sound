# python main.py --task task_level --wav wav_path --in input_json --out output_json

# Imports
import os
import sys
import getopt
import argparse
import librosa
import librosa.display
import numpy as np
import torch
import cv2
import json
import torch.nn as nn
import matplotlib.pyplot as plt
import models.model_1_1 as model_1_1
import models.model_1_2 as model_1_2
import models.model_2_1 as model_2_1
import models.model_2_2 as model_2_2


use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")


def read_process_audio_file_event_level(audio_file_path, annotation_file):

    y, sr = librosa.load(audio_file_path)
    num = 0
    for event in annotation_file["event_annotation"]:
        s = int(event["start"])*sr//1000
        e = int(event["end"])*sr//1000
        e_type = event["type"]
        y_segment = y[s:e]

        folder_path = os.path.join("temp", "event")
        os.makedirs(folder_path, exist_ok = True)

        D = np.abs(librosa.stft(y_segment))**2
        S = librosa.feature.melspectrogram(S=D, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000)
            
        plt.axis("off")
        plt.savefig(os.path.join(folder_path,"mel.png"))
        plt.close()

def read_process_audio_file_record_level(audio_file_path, annotation_file):

    y, sr = librosa.load(audio_file_path)

    r_type = annotation_file["record_annotation"]

    folder_path = os.path.join("temp", "record")
    os.makedirs(folder_path, exist_ok = True)

    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000)
        
    plt.axis("off")
    plt.savefig(os.path.join(folder_path,"mel.png"))
    plt.close()


def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']



# Parsing arguments
try:
  argv = sys.argv[1:]
except:
  print("error")
# print(argv)

task_level = argv[1]
wav_path = argv[3]
input_json_path = argv[5]
output_json_path = argv[7]

f = open(input_json_path)
f = json.load(f)
print(f)

if task_level=="event":
  labels = ['Coarse Crackle', 'Fine Crackle', 'Normal', 'Stridor', 'Wheeze']
  read_process_audio_file_event_level(wav_path, f)
  checkpoint_path = "checkpoint/checkpoint_20_1_2.pt"

  model = model_1_2.Classifier1_2()
  model = model.to(device)
  model = nn.DataParallel(model)
  model_opt = torch.optim.SGD(model.parameters(),lr=0.01)
  model, model_opt, __ = load_ckp(checkpoint_path, model, model_opt)

  src_path = os.path.join("temp", "event", "mel.png")
  src = cv2.imread(src_path)
  src = cv2.resize(src, (224, 224))
  src = np.array(src)
  src = np.transpose(src,(2, 0, 1))
  src = torch.FloatTensor(src)
  src = src.unsqueeze(0)
  src = torch.concat([src, src],dim=0)
  out = model.forward(src.to(device))
  pred = torch.argmax(out, dim=1).cpu().numpy()

elif task_level=="record":
  labels = ['CAS', 'CAS _ DAS', 'DAS', 'Normal', 'Poor Quality']
  read_process_audio_file_record_level(wav_path, f)
  checkpoint_path = "checkpoint/checkpoint_20_2_2.pt"

  model = model_2_2.Classifier2_2()
  model = model.to(device)
  model = nn.DataParallel(model)
  model_opt = torch.optim.SGD(model.parameters(),lr=0.01)
  model, model_opt, __ = load_ckp(checkpoint_path, model, model_opt)

  src_path = os.path.join("temp", "record", "mel.png")
  src = cv2.imread(src_path)
  src = cv2.resize(src, (224, 224))
  src = np.array(src)
  src = np.transpose(src,(2, 0, 1))
  src = torch.FloatTensor(src)
  src = src.unsqueeze(0)
  src = torch.concat([src, src],dim=0)
  out = model.forward(src.to(device))
  pred = torch.argmax(out, dim=1).cpu().numpy()

print(labels[pred[0]])
