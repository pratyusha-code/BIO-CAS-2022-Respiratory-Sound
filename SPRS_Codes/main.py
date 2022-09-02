import os
from matplotlib import pyplot as plt
import numpy as np
import argparse

import torch
from torch import cuda
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, FrequencyMasking
from tqdm import tqdm
import cv2
from PIL import Image

from utils import cuda_gpu, plot_spectrogram_test, imshow, plot_dataset, model_summary
from data_loader import data_loader, test_data_loader
from model_script import get_pretrained_model
from test import test_model, test


def main_func(task_level, wav_path, out_path, train_on_gpu, multi_gpu, device):
    
    main_path = wav_path
    main_path = main_path.split('/')[0:-2]
    main_path.append('')
    main_path = '/'.join(main_path)

    processed_dir = main_path+"processed/"
    os.makedirs(processed_dir, exist_ok=True)    
    wav_files = os.listdir(wav_path)

    with open(out_path,"w") as outfile:
        outfile.write("{")

    for i in tqdm(range(len(wav_files))):
        y, sr = torchaudio.load(os.path.join(wav_path, wav_files[i]))

        S = MelSpectrogram(sample_rate=sr, n_fft=2048, win_length=128, hop_length=32, pad=1, n_mels=128, mel_scale="htk", pad_mode="reflect", norm="slaney")(y) 
        S_dB = AmplitudeToDB(stype='power')(S)
        plot_spectrogram_test(S_dB, i, processed_dir)

        print(os.path.join(processed_dir, str(i)+".png"))

        spec = cv2.imread(os.path.join(processed_dir, str(i)+".png"))
        print(spec.shape)
        spec = cv2.resize(spec, (224, 224))

        # transform = transforms.Compose([transforms.Resize(255),
		# 							transforms.CenterCrop(224),
		# 							transforms.ToTensor()])

        # spec = Image.fromarray(spec)

        # spec = transform(spec)

        if task_level=='11' or task_level=='12':
#             Task 1
            categories = ['Normal', 'Ronchi', 'Wheeze', 'Stridor', 'Coarse Crackle', 'Fine Crackle', 'Wheeze & Crackle']
            n_classes = len(categories)
            batch_size = 1
            # test_dataloader = test_data_loader(batch_size, processed_dir)
            n_classes = len(categories)
            model_name = 'vgg16'
            # model_name = 'resnet50'
            # model_name = 'mobilenet'

            model = get_pretrained_model(model_name, n_classes, multi_gpu, train_on_gpu, device)

            criterion = nn.CrossEntropyLoss()
            criterion.cuda()
            if multi_gpu:
                optimizer = optim.Adam(model.module.classifier.parameters(), lr=0.003)
            elif train_on_gpu:
                optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
            save_path = "models/model_event.pt"

            label = test(model_name, model, save_path, spec, task_level, optimizer, criterion, device)

            # j_write = {
            #     wav_files[i]:label
            # }


            # with open(out_path,"a") as outfile:
            #     outfile.write(str(j_write))
            if i==0:
                if task_level=='11':
                    with open(out_path,"a") as outfile:
                        if categories[label]=='Normal':
                            outfile.write("\n\t"+str(wav_files[i])+":"+str(categories[label]))
                        else:
                            outfile.write("\n\t"+str(wav_files[i])+":"+"Adventitious")
                elif task_level=='12':
                    with open(out_path,"a") as outfile:
                        outfile.write("\n\t"+str(wav_files[i])+":"+str(categories[label]))
            else:
                if task_level=='11':
                    with open(out_path,"a") as outfile:
                        if categories[label]=='Normal':
                            outfile.write(",\n\t"+str(wav_files[i])+":"+str(categories[label]))
                        else:
                            outfile.write(",\n\t"+str(wav_files[i])+":"+"Adventitious")
                elif task_level=='12':
                    with open(out_path,"a") as outfile:
                        outfile.write(",\n\t"+str(wav_files[i])+":"+str(categories[label]))

        elif task_level=='21' or task_level=='22':
#             Task 2
            categories = ['Normal', 'CAS', 'DAS', 'CAS & DAS', 'Poor Quality']
            n_classes = len(categories)
            batch_size = 1

            # test_dataloader = test_data_loader(batch_size, processed_dir)
            model_name = 'vgg16'
            model = get_pretrained_model(model_name, n_classes, multi_gpu, train_on_gpu, device)

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

            save_path = "models/model_record.pt"

            label = test(model_name, model, save_path, spec, task_level, optimizer, criterion, device)

            # j_write = {
            #     wav_files[i]:label
            # }

            # print(j_write)
            if i==0:
                if task_level=='21':
                    with open(out_path,"a") as outfile:
                        if categories[label]=='Normal':
                            outfile.write("\n\t"+str(wav_files[i])+":"+str(categories[label]))
                        elif categories[label]=='Poor Quality':
                            outfile.write("\n\t"+str(wav_files[i])+":"+str(categories[label]))
                        else:
                            outfile.write("\n\t"+str(wav_files[i])+":"+"Adventitious")
                elif task_level=='22':
                    with open(out_path,"a") as outfile:
                        outfile.write("\n\t"+str(wav_files[i])+":"+str(categories[label]))
            else:
                if task_level=='21':
                    with open(out_path,"a") as outfile:
                        if categories[label]=='Normal':
                            outfile.write(",\n\t"+str(wav_files[i])+":"+str(categories[label]))
                        elif categories[label]=='Poor Quality':
                            outfile.write(",\n\t"+str(wav_files[i])+":"+str(categories[label]))
                        else:
                            outfile.write(",\n\t"+str(wav_files[i])+":"+"Adventitious")
                elif task_level=='22':
                    with open(out_path,"a") as outfile:
                        outfile.write(",\n\t"+str(wav_files[i])+":"+str(categories[label]))
            # with open(out_path,"a") as outfile:
            #     outfile.write("\n\t"+str(wav_files[i])+":"+str(categories[label])+",")

    with open(out_path,"a") as outfile:
        outfile.write("\n}")

host = 'ada'
train_on_gpu, multi_gpu, device = cuda_gpu(host)

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, type=str)
parser.add_argument('--wav', required=True, type=str)
parser.add_argument('--out', required=True, type=str)
args = parser.parse_args()

print(args[0],args[1],args[2])

task_level = args[0]
wav_path = args[1]
out_path = args[2]
main_func(task_level, wav_path, out_path, train_on_gpu, multi_gpu, device)
