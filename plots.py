import pickle
import numpy as np
import argparse
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Number of workers for dataloader
workers = 6

# Root directory for dataset
dataroot = "/home/berkan/Desktop/Repo/BanditsGetGANs/celeba/"

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

####################################################
# Load and standardize data
# Original GAN
with open('G_loss_orig', 'rb') as f:
    G_losses_orig = pickle.load(f)
f.close()
with open('D_loss_orig', 'rb') as f:
    D_losses_orig = pickle.load(f)
f.close()
with open('test_imgs_orig', 'rb') as f:
    img_list_orig = pickle.load(f)
f.close()

G_orig_mean = np.mean(G_losses_orig, axis=0)
G_orig_std = np.std(G_losses_orig - G_orig_mean, axis=0)
D_orig_mean = np.mean(D_losses_orig, axis=0)
D_orig_std = np.std(D_losses_orig - D_orig_mean, axis=0)
####################################################
# MAB GAN
stat_reward = False
conf_bound = False
with open('G_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    G_losses_MAB_stat_false_conf_false = pickle.load(f)
f.close()
with open('D_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    D_losses_MAB_stat_false_conf_false = pickle.load(f)
f.close()
with open('test_imgs_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    img_list_MAB_stat_false_conf_false = pickle.load(f)
f.close()

G_losses_MAB_stat_false_conf_false_mean = np.mean(G_losses_MAB_stat_false_conf_false, axis=0)
G_losses_MAB_stat_false_conf_false_std = np.std(G_losses_MAB_stat_false_conf_false - G_losses_MAB_stat_false_conf_false_mean, axis=0)
D_losses_MAB_stat_false_conf_false_mean = np.mean(D_losses_MAB_stat_false_conf_false, axis=0)
D_losses_MAB_stat_false_conf_false_std = np.std(D_losses_MAB_stat_false_conf_false - D_losses_MAB_stat_false_conf_false_mean, axis=0)
####################################################
stat_reward = True
conf_bound = False
with open('G_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    G_losses_MAB_stat_true_conf_false = pickle.load(f)
f.close()
with open('D_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    D_losses_MAB_stat_true_conf_false = pickle.load(f)
f.close()
with open('test_imgs_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    img_list_MAB_stat_true_conf_false = pickle.load(f)
f.close()

G_losses_MAB_stat_true_conf_false_mean = np.mean(G_losses_MAB_stat_true_conf_false, axis=0)
G_losses_MAB_stat_true_conf_false_std = np.std(G_losses_MAB_stat_true_conf_false - G_losses_MAB_stat_true_conf_false_mean, axis=0)
D_losses_MAB_stat_true_conf_false_mean = np.mean(D_losses_MAB_stat_true_conf_false, axis=0)
D_losses_MAB_stat_true_conf_false_std = np.std(D_losses_MAB_stat_true_conf_false - D_losses_MAB_stat_true_conf_false_mean, axis=0)
####################################################
stat_reward = False
conf_bound = True
with open('G_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    G_losses_MAB_stat_false_conf_true = pickle.load(f)
f.close()
with open('D_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    D_losses_MAB_stat_false_conf_true = pickle.load(f)
f.close()
with open('test_imgs_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    img_list_MAB_stat_false_conf_true = pickle.load(f)
f.close()

G_losses_MAB_stat_false_conf_true_mean = np.mean(G_losses_MAB_stat_false_conf_true, axis=0)
G_losses_MAB_stat_false_conf_true_std = np.std(G_losses_MAB_stat_false_conf_true - G_losses_MAB_stat_false_conf_true_mean, axis=0)
D_losses_MAB_stat_false_conf_true_mean = np.mean(D_losses_MAB_stat_false_conf_true, axis=0)
D_losses_MAB_stat_false_conf_true_std = np.std(D_losses_MAB_stat_false_conf_true - D_losses_MAB_stat_false_conf_true_mean, axis=0)
####################################################
stat_reward = True
conf_bound = True
with open('G_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    G_losses_MAB_stat_true_conf_true = pickle.load(f)
f.close()
with open('D_loss_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    D_losses_MAB_stat_true_conf_true = pickle.load(f)
f.close()
with open('test_imgs_MAB'+'_stat_'+str(stat_reward)+'_ucb_'+str(conf_bound), 'rb') as f:
    img_list_MAB_stat_true_conf_true = pickle.load(f)
f.close()

G_losses_MAB_stat_true_conf_true_mean = np.mean(G_losses_MAB_stat_true_conf_true, axis=0)
G_losses_MAB_stat_true_conf_true_std = np.std(G_losses_MAB_stat_true_conf_true - G_losses_MAB_stat_true_conf_true_mean, axis=0)
D_losses_MAB_stat_true_conf_true_mean = np.mean(D_losses_MAB_stat_true_conf_true, axis=0)
D_losses_MAB_stat_true_conf_true_std = np.std(D_losses_MAB_stat_true_conf_true - D_losses_MAB_stat_true_conf_true_mean, axis=0)
####################################################
# Generator Plots
plt.figure(figsize=(10,5))
plt.title("Generator Loss During Training")
plt.fill_between(range(len(G_orig_mean)), y1=G_orig_mean-G_orig_std, y2=G_orig_mean+G_orig_std, label="Standard GAN")
plt.fill_between(range(len(G_losses_MAB_stat_true_conf_true_mean)),
                 y1=G_losses_MAB_stat_true_conf_true_mean-G_losses_MAB_stat_true_conf_true_std,
                 y2=G_losses_MAB_stat_true_conf_true_mean+G_losses_MAB_stat_true_conf_true_std, label="MAB GAN stat. and UCB")
plt.fill_between(range(len(G_losses_MAB_stat_false_conf_true_mean)),
                 y1=G_losses_MAB_stat_false_conf_true_mean-G_losses_MAB_stat_false_conf_true_std,
                 y2=G_losses_MAB_stat_false_conf_true_mean+G_losses_MAB_stat_false_conf_true_std, label="MAB GAN non-stat. and UCB")
plt.fill_between(range(len(G_losses_MAB_stat_true_conf_false_mean)),
                 y1=G_losses_MAB_stat_true_conf_false_mean-G_losses_MAB_stat_true_conf_false_std,
                 y2=G_losses_MAB_stat_true_conf_false_mean+G_losses_MAB_stat_true_conf_false_std, label="MAB GAN stat. and no UCB")
plt.fill_between(range(len(G_losses_MAB_stat_false_conf_false_mean)),
                 y1=G_losses_MAB_stat_false_conf_false_mean-G_losses_MAB_stat_false_conf_false_std,
                 y2=G_losses_MAB_stat_false_conf_false_mean+G_losses_MAB_stat_false_conf_false_std, label="MAB GAN non-stat. and no UCB")
plt.xlabel("iterations")
plt.ylabel("Generator Loss")
plt.legend()
plt.show()
####################################################
# Discriminator Plots
plt.figure(figsize=(10,5))
plt.title("Discriminator Loss During Training")
plt.fill_between(range(len(D_orig_mean)), y1=D_orig_mean-D_orig_std, y2=D_orig_mean+D_orig_std, label="Standard GAN")
plt.fill_between(range(len(D_losses_MAB_stat_true_conf_true_mean)),
                 y1=D_losses_MAB_stat_true_conf_true_mean-D_losses_MAB_stat_true_conf_true_std,
                 y2=D_losses_MAB_stat_true_conf_true_mean+D_losses_MAB_stat_true_conf_true_std, label="MAB GAN stat. and UCB")
plt.fill_between(range(len(D_losses_MAB_stat_false_conf_true_mean)),
                 y1=D_losses_MAB_stat_false_conf_true_mean-D_losses_MAB_stat_false_conf_true_std,
                 y2=D_losses_MAB_stat_false_conf_true_mean+D_losses_MAB_stat_false_conf_true_std, label="MAB GAN non-stat. and UCB")
plt.fill_between(range(len(D_losses_MAB_stat_true_conf_false_mean)),
                 y1=D_losses_MAB_stat_true_conf_false_mean-D_losses_MAB_stat_true_conf_false_std,
                 y2=D_losses_MAB_stat_true_conf_false_mean+D_losses_MAB_stat_true_conf_false_std, label="MAB GAN stat. and no UCB")
plt.fill_between(range(len(D_losses_MAB_stat_false_conf_false_mean)),
                 y1=D_losses_MAB_stat_false_conf_false_mean-D_losses_MAB_stat_false_conf_false_std,
                 y2=D_losses_MAB_stat_false_conf_false_mean+D_losses_MAB_stat_false_conf_false_std, label="MAB GAN non-stat. and no UCB")
plt.xlabel("iterations")
plt.ylabel("Discriminator Loss")
plt.legend()
plt.show()

####################################################
# **Real Images vs. Fake Images**
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the fake images from the last epoch
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Fake Images for Standard GAN Training")
plt.imshow(np.transpose(img_list_orig[-1],(1,2,0)))
plt.show()

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images for MAB GAN Training")
plt.imshow(np.transpose(img_list_MAB_stat_true_conf_true[-1],(1,2,0)))
plt.show()

######################################################################
# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# **Visualization of G’s progression**
#
# Remember how we saved the generator’s output on the fixed_noise batch
# after every epoch of training. Now, we can visualize the training
# progression of G with an animation. Press the play button to start the
# animation.
#

#%%capture
#fig = plt.figure(figsize=(8,8))
#plt.axis("off")
#ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
#ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

#HTML(ani.to_jshtml())
