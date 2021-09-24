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
from net import UNet

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
# if os.path.isfile():  # continue training from old weights
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

torch.save(model, 'model_brats_domain.pt')  # save weights 

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
    plt.show()
    plt.savefig('show_results.png')

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
