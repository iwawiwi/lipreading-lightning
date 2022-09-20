# append working directory to PATH system
import os
import sys

import torch
import torchvision.transforms as transforms
from imgcaption_loader import get_loader
from PIL import Image

from src.datamodules.components.flickr8k import Flickr8kDataset
from src.models.imgcaption_module import Flickr8KLitModule

sys.path.append("/home/iwawiwi/research/22/lipreading-lightning")


def print_examples(
    model,
    device,
    vocab,
    sample_root="/home/iwawiwi/research/22/lipreading-lightning/src/vendor/example",
):
    # define transformation to input sample
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    model.to(device)  # move model to device
    test_img1 = transform(
        Image.open(os.path.join(sample_root, "dog.jpg")).convert("RGB")
    ).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print("Example 1 OUTPUT: " + " ".join(model.caption_image(test_img1.to(device), vocab)))
    test_img2 = transform(
        Image.open(os.path.join(sample_root, "child.jpg")).convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print("Example 2 OUTPUT: " + " ".join(model.caption_image(test_img2.to(device), vocab)))
    test_img3 = transform(
        Image.open(os.path.join(sample_root, "bus.png")).convert("RGB")
    ).unsqueeze(0)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print("Example 3 OUTPUT: " + " ".join(model.caption_image(test_img3.to(device), vocab)))
    test_img4 = transform(
        Image.open(os.path.join(sample_root, "boat.png")).convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print("Example 4 OUTPUT: " + " ".join(model.caption_image(test_img4.to(device), vocab)))
    test_img5 = transform(
        Image.open(os.path.join(sample_root, "horse.png")).convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: " + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )

    return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/home/iwawiwi/research/22/lipreading-lightning/data/flickr8k"
BEST_MODEL = "/home/iwawiwi/research/22/lipreading-lightning/src/vendor/model.pth"

loader, dataset = get_loader(
    root_folder=DATA_DIR + "/Images", annotation_file=DATA_DIR + "/captions.txt", transform=None
)

# load from lightning training module (DO NOT directly load as Model class)
trained_model = Flickr8KLitModule.load_from_checkpoint(checkpoint_path=BEST_MODEL)
# print loaded model hparams
# print(trained_model.hparams)

# switch to evaluation mode
trained_model.eval()
trained_model.freeze()

# load vocabulary from dataset
dataset = Flickr8kDataset(
    root_dir=DATA_DIR + "/Images",
    captions_file=DATA_DIR + "/captions.txt",
)

# DO CAPTIONING
# model stored in ```trained_model.net``` or ```trained_model.hparams.net```
print_examples(trained_model.net, device, dataset.vocab)
