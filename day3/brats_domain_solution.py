"""
This file provides a solution for brats_domain by Jingnan Jia.
You can directly run this file on command.
"""
import random
from tqdm import trange
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import Dataset, DataLoader, partition_dataset
from monai.networks import eval_mode
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    rescale_array,
    ScaleIntensityd,
)
from monai.utils import set_determinism
import torchvision.models as models

# ------This block is for UNet building, by the author: Jingnan Jia ------------------#
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ------The above block is for UNet building ------------------#


print_config()
set_determinism(0)

import tempfile

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

data_dir = os.path.join(root_dir, "brain_2d")
resource = "https://drive.google.com/uc?id=17f4J_rU5pi1zRmxMe5OwljyT3tlBf6qI"
compressed_file = os.path.join(root_dir, "brain_2d.tar.gz")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir)

input_ims = sorted(glob(os.path.join(data_dir, "*input.npy")))
output_ims = sorted(glob(os.path.join(data_dir, "*GT_output.npy")))
data = [{"input": i, "output": o} for i, o in zip(input_ims, output_ims)]
print("number data points", len(data))
print("example", data[0])


class ChannelWiseScaleIntensityd(MapTransform):
    """Perform channel-wise intensity normalisation."""

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, d):
        for key in self.keys:
            for idx, channel in enumerate(d[key]):
                d[key][idx] = rescale_array(channel)
        return d


keys = ["input", "output"]
train_transforms = Compose([
    LoadImaged(keys),
    ChannelWiseScaleIntensityd("input"),
    ScaleIntensityd("output"),
    EnsureTyped(keys),
])
val_transforms = Compose([
    LoadImaged(keys),
    ChannelWiseScaleIntensityd("input"),
    ScaleIntensityd("output"),
    EnsureTyped(keys),
])

t = train_transforms(data[0])
print(t["input"].shape, t["output"].shape)
in_channels, out_channels = t["input"].shape[0], t["output"].shape[0]

# split data into 80% and 20% for training and validation, respectively
train_data, val_data = partition_dataset(data, (8, 2), shuffle=True)
print("num train data points:", len(train_data))
print("num val data points:", len(val_data))
batch_size = 20
num_workers = 10
train_ds = Dataset(train_data, train_transforms)
train_dl = DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True)
val_ds = Dataset(val_data, val_transforms)
val_dl = DataLoader(val_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------The following part is filled by the author: Jingnan Jia -------
# Create loss fn and optimiser
max_epochs = 1  # TODO
model = UNet(3, 1)  # TODO
model_fpath = 'model_brats_domain.pt'
save_model_flag = True  # if save model weights to disk
# if save_model_flag and os.path.isfile():  # continue training from old weights
#     model.load_state_dict(torch.load())

model.to(device)  # TODO
loss_function = torch.nn.MSELoss()  # TODO
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  # TODO
# -----------------------------------------------------------------------------


epoch_losses = []

t = trange(max_epochs, desc=f"epoch 0, avg loss: inf", leave=True)
for epoch in t:
    model.train()
    epoch_loss = 0
    step = 0
    for batch in train_dl:
        step += 1
        inputs, outputs_gt = batch["input"].to(device), batch["output"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, outputs_gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    epoch_losses.append(epoch_loss)
    t.set_description(f"epoch {epoch + 1}, average loss: {epoch_loss:.4f}")

if save_model_flag:  # save weights
    torch.save(model, 'model_brats_domain.pt')

plt.plot(epoch_losses)


def imshows(ims):
    """Visualises a list of dictionaries.
    Each key of the dictionary will be used as a column, and
    each element of the list will be a row.
    """
    nrow = len(ims)
    ncol = len(ims[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(
        ncol * 3, nrow * 3), facecolor='white')
    for i, im_dict in enumerate(ims):
        for j, (title, im) in enumerate(im_dict.items()):
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy()
            # If RGB, put to end. Else, average across channel dim
            if im.ndim > 2:
                im = np.moveaxis(im, 0, -1) if im.shape[0] == 3 else np.mean(im, axis=0)

            ax = axes[j] if len(ims) == 1 else axes[i, j]
            ax.set_title(f"{title}\n{im.shape}")
            im_show = ax.imshow(im)
            ax.axis("off")
    # plt.show()  # show figure
    plt.savefig('compare_results.png')  # save figure


to_imshow = []

_ = model.eval()

for idx in np.random.choice(len(val_ds), size=5, replace=False):
    rand_data = val_ds[idx]
    rand_input, rand_output_gt = rand_data["input"], rand_data["output"]
    rand_output = model(rand_input.to(device)[None])[0]
    to_imshow.append(
        {
            "FLAIR": rand_input[0],
            "T1w": rand_input[1],
            "T2w": rand_input[2],
            "GT GD": rand_output_gt,
            "inferred GD": rand_output,
        }
    )
imshows(to_imshow)
