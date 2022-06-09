import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from imgcaption_loader import get_loader
from imgcaption_model import CNNtoRNN


def train():

    # define the transform
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # dataloader
    trainloader, dataset = get_loader(
        root_folder="/home/iwawiwi/research/22/lipreading-lightning/data/flickr8k/Images",
        annotation_file="/home/iwawiwi/research/22/lipreading-lightning/data/flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Train Using {device}")

    # Hyperparameter
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    print(vocab_size)
    num_layers = 1
    lrate = 0.001
    num_epochs = 5

    # Initialize Model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    print(dataset.vocab.stoi["<PAD>"])
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    model.train()

    for epoch in range(num_epochs):

        for idx, (imgs, captions) in enumerate(trainloader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if idx % 100 == 0:
                print(
                    "Epoch: {}/{}".format(epoch + 1, num_epochs),
                    "Step: {}".format(idx),
                    "Loss: {}".format(loss.item()),
                )

            break


if __name__ == "__main__":
    train()
