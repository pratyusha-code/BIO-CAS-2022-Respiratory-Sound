import os
import numpy as np
import torch
from torch import cuda
import matplotlib.pyplot as plt
from torchsummary import summary

def cuda_gpu(host):
    """Adding cuda modules and checking gpu count"""
    
    # if host=='ada':
    #     os.system("module add u18/cuda/10.0")
    #     os.system("module add u18/cudnn/7.6-cuda-10.0")

    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return train_on_gpu, multi_gpu, device


def plot_spectrogram_train_events(spec, t, i, j ,k, l, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    folder_path = os.path.join(processed_dir, t)
    os.makedirs(folder_path, exist_ok = True)

    plt.figure()
    # im = plt.imshow(spec.reshape([spec.size()[1], spec.size()[2], spec.size()[0]]), origin="lower", aspect="auto")
    im = plt.imshow(spec, origin="lower", aspect="auto")
    plt.axis("off")
    plt.savefig(os.path.join(folder_path,str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l)+".png"))
    plt.close()


def plot_spectrogram_train_records(spec, t, i, l, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    folder_path = os.path.join(processed_dir, t)
    os.makedirs(folder_path, exist_ok = True)

    plt.figure()
    # im = plt.imshow(spec.reshape([spec.size()[1], spec.size()[2], spec.size()[0]]), origin="lower", aspect="auto")
    im = plt.imshow(spec, origin="lower", aspect="auto")
    plt.axis("off")
    plt.savefig(os.path.join(folder_path, str(i) + "_" + str(l)+".png"))
    plt.close()


def plot_spectrogram_test(spec, i, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

    plt.figure()
    im = plt.imshow(spec.reshape([spec.size()[1], spec.size()[2], spec.size()[0]]), origin="lower", aspect="auto")
    # im = plt.imshow(spec, origin="lower", aspect="auto")
    plt.show()
    plt.axis("off")
    plt.savefig(os.path.join(processed_dir, str(i)+".png"))
    plt.close()

# def plot_spectrogram(spec, i, t, processed_dir, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
#     os.makedirs(processed_dir, exist_ok=True)
#     folder_path = os.path.join(processed_dir, t)
#     os.makedirs(folder_path, exist_ok = True)

#     plt.figure()
#     im = plt.imshow(spec.reshape([spec.size()[1], spec.size()[2], spec.size()[0]]), origin="lower", aspect="auto")
#     plt.show()
#     plt.axis("off")
#     plt.savefig(os.path.join(processed_dir, str(i)+".png"))
#     plt.close()


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()

    return ax


def plot_dataset(dataloader, r, c):
    data_iter = iter(dataloader)

    fig, axes = plt.subplots(figsize=(10,10), nrows=r, ncols=c)
    for row in range(r):
        for col in range(c):
            images, labels = next(data_iter)
            ax = axes[row, col]
            imshow(images[0], ax=ax, normalize=False)
            ax.title.set_text(dataloader.dataset.classes[labels[0]])
    plt.show()
    return


def model_summary(model, multi_gpu, input_size, batch_size, device):
	if multi_gpu:
		summary(
			model.module,
			input_size=input_size,
			batch_size=batch_size,
			device=device)
	else:
		summary(
			model,
			input_size=input_size,
			batch_size=batch_size,
			device=device)
	return

