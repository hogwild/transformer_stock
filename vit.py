import numpy as np
import pickle as pk
import random

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.utils import save_image
# from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def slicing(images, n_patches, candle_width=3):
    n, c, h, w = images.shape
    assert candle_width*n_patches == w, "Candle width * npatches should be equal to image width"
    patches = torch.zeros(n, n_patches, candle_width*h)
    for idx, image in enumerate(images):
        for i in range(n_patches):
            patch = image[:, :, i*candle_width: (i+1)*candle_width]
            patches[idx, i] = patch.flatten()
    return patches

def visualization(slicing, img_height, candle_width, n_chanels=1):
    n, n_patches, candle_pixels = slicing.shape
    images = torch.zeros(n, n_chanels, img_height, candle_width*n_patches )
    for idx, image in enumerate(slicing):
        for i in range(n_patches-1):
            patch = torch.reshape(slicing[idx][i], (img_height, candle_width))
            # print(patch.shape, i)
            images[idx, :, :, i*candle_width:(i+1)*candle_width] = patch
    return images



class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for i in range(len(Q)):
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq_q = Q[i][:, head * self.d_head: (head + 1) * self.d_head]
                seq_k = K[i][:, head * self.d_head: (head + 1) * self.d_head]
                seq_v = V[i][:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq_q), k_mapping(seq_k), v_mapping(seq_v)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5)) ##here @ is the operator of matrix multiplation
                if mask is not None:
                    # print("attention device", attention.get_device())
                    attention = attention.masked_fill(mask == 0, -1e9)
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        result = torch.cat([torch.unsqueeze(r, dim=0) for r in result])
        # print("attention shape", result.shape, seq_result[0].shape)
        # print("q, k, v shape", q.shape, k.shape, v.shape)
        # if mask is not None:
        #     print("mask shape", mask.shape)
        return result


class MyEncoderViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyEncoderViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.self_attn = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        norm1_x = self.norm1(x)
        out = x + self.self_attn(norm1_x, norm1_x, norm1_x)
        # norm2_out = self.norm2(out)
        out = out + self.mlp(self.norm2(out))
        return out


class MyDecoderViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyDecoderViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.self_attn = MyMSA(hidden_d, n_heads)
        self.cross_attn = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x, enc_output):
        norm1_x = self.norm1(x)
        # print('x shape', x.shape)
        out = x + self.self_attn(norm1_x, norm1_x, norm1_x, self.tgt_mask)
        norm1_enc_output = self.norm1(enc_output)
        out = out + self.cross_attn(out, norm1_enc_output, norm1_enc_output)
        out = out + self.mlp(self.norm2(out))
        return out


class MyViTAutoEncoder(nn.Module):
    def __init__(self, src_chw, tgt_chw, n_src_patches=20, n_tgt_patches=5, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViTAutoEncoder, self).__init__()
        
        # Attributes
        self.src_chw = src_chw # ( C , H , W )
        self.tgt_chw = tgt_chw
        self.n_src_patches = n_src_patches
        self.n_tgt_patches = n_tgt_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        # assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert src_chw[2] % n_src_patches == 0, "Input source shape not entirely divisible by number of patches"
        assert tgt_chw[2] % n_tgt_patches == 0, "Input target shape not entirely divisible by number of patches"
        self.src_patch_size = (src_chw[1], src_chw[2] / n_src_patches)
        self.tgt_patch_size = (tgt_chw[1], tgt_chw[2] / n_tgt_patches)

        # 1) Linear mapper
        self.input_d = int(src_chw[0] * self.src_patch_size[0] * self.src_patch_size[1])
        self.encoder_linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        self.output_d = int(tgt_chw[0] * self.tgt_patch_size[0] * self.tgt_patch_size[1])
        self.decoder_linear_mapper = nn.Linear(self.output_d, self.hidden_d)
        print("input d and outout d", self.input_d, self.output_d)
        # 2) Learnable classification token
        # self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('src_positional_embeddings', get_positional_embeddings(n_src_patches, hidden_d), persistent=False)
        self.register_buffer('tgt_positional_embeddings', get_positional_embeddings(n_tgt_patches, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([MyEncoderViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([MyDecoderViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 6) Classification MLPk
        self.mlp = nn.Linear(self.hidden_d, self.output_d)

        # 7) reshape to image
            

    def forward(self, src_imgs, tgt_imgs):
        # Dividing images into patches
        # candle width = 3
        n, c, h_tgt_img, w = tgt_imgs.shape
        src_patches = slicing(src_imgs, self.n_src_patches, 3).to(self.src_positional_embeddings.device)
        tgt_patches = slicing(tgt_imgs, self.n_tgt_patches, 3).to(self.tgt_positional_embeddings.device)

        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        src_tokens = self.encoder_linear_mapper(src_patches)
        tgt_tokens = self.decoder_linear_mapper(tgt_patches)
        # print(src_tokens.shape, tgt_tokens.shape)
        # Adding classification token to the tokens
        # tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        # print(self.tgt_positional_embeddings.repeat(n, 1, 1).shape)
        # Adding positional embedding
        src_out = src_tokens + self.src_positional_embeddings.repeat(n, 1, 1)
        tgt_out = tgt_tokens + self.tgt_positional_embeddings.repeat(n, 1, 1)
        # tgt_mask = self.generate_mask(tgt_out)
        # print(src_out.shape, tgt_out.shape)
        # Transformer Blocks
        # src_mask, tgt_mask = self.generate_mask(src_out, tgt_out)
        # tgt_mask.to(device)
        # print("the device", device)
        # print("tgt_mask device", tgt_mask.get_device())
        for block in self.encoder_blocks:
            src_out = block(src_out)
        
        for block in self.decoder_blocks:
            tgt_out = block(tgt_out, src_out, self.tgt_mask)
            
        # Getting the classification token only
        # out = out[:, 0]
        # print(tgt_out.shape)
        slicing = self.mlp(tgt_out)

        return self.visualization(slicing, h_tgt_img)
    
    def generate_mask(self, tgt_imgs_size, candle_width=3):
        n, c, h, w = tgt_imgs_size
        tgt_patches = torch.zeros(n, self.n_tgt_patches, candle_width*h).to(self.tgt_positional_embeddings.device)
        tgt_tokens = self.decoder_linear_mapper(tgt_patches)
        seq_length = tgt_tokens.size(1)
        self.tgt_mask = (1 - torch.triu(torch.ones(seq_length, seq_length), diagonal=1)).bool()
    
    def visualization(self, slicing, img_height, candle_width=3, n_chanels=1):
        n, n_patches, candle_pixels = slicing.shape
        images = torch.zeros(n, n_chanels, img_height, candle_width*n_patches )
        for idx, image in enumerate(slicing):
            for i in range(n_patches-1):
                patch = torch.reshape(slicing[idx][i], (img_height, candle_width))
                # print(patch.shape, i)
                images[idx, :, :, i*candle_width:(i+1)*candle_width] = patch
        return images
        # return src_mask, tgt_mask

    # def generate_mask(self, src, tgt):
    #     # src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     # tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    #     # seq_length = tgt.size(1)
    #     # nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    #     # tgt_mask = tgt_mask & nopeak_mask
    #     src_mask = None
    #     seq_length = tgt.size(1)
    #     # tgt_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    #     tgt_mask = (1 - torch.triu(torch.ones(seq_length, seq_length), diagonal=1)).bool().to(device)
    #     # tgt_mask.to(device)
    #     return src_mask, tgt_mask


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


# def main():
#     # Loading data
#     # transform = ToTensor()

    
#     # train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
#     # test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
#     import pickle as pk
#     import torchvision.transforms as transforms
#     # train_set = MNIST(root='./../datasets', train=True, download=True, transform=transforms.ToTensor())
#     # test_set = MNIST(root='./../datasets', train=False, download=True, transform=transforms.ToTensor())
#     # print(train_set.shape)

#     symbols = ["601857"]

#     years = range(2008,2022)
#     imgs = []
#     labels = []
#     for s in symbols:
#         for y in years:
#             f = open(f"./kchart/{s}_{y}_imgs.pk", "rb")
#             imgs.extend((pk.load(f)))
#             f = open(f"./kchart/{s}_{y}_labels.pk", "rb")
#             labels.extend(pk.load(f))
    
#     convert_tensor = ToTensor()
#     # print(convert_tensor(imgs[0]).size())
#     imgs = [convert_tensor(img) for img in imgs]
#     labels = [convert_tensor(img) for img in labels]
#     data = list(zip(imgs, labels))
#     # print(imgs[0].shape, labels[0].shape)
#     idx_train = round(0.8*(len(data)))
#     train_set = data[:idx_train]
#     test_set = data[idx_train:]
#     train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
#     test_loader = DataLoader(test_set, shuffle=False, batch_size=64)

#     # Defining model and training options
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
#     model = MyViTAutoEncoder((1, 64, 60), (1, 32, 15), n_src_patches=20, n_tgt_patches=5, n_blocks=8, hidden_d=8, n_heads=2, out_d=10).to(device)
#     N_EPOCHS = 1000
#     LR = 0.005

#     # Training loop
#     optimizer = Adam(model.parameters(), lr=LR)
#     criterion = CrossEntropyLoss()
#     for epoch in trange(N_EPOCHS, desc="Training"):
#         train_loss = 0.0
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
#             x, y = batch
#             x, y = x.to(device), y.to(device)
#             y_hat = model(x, y)

#             y = slicing(y, 5, 3).to(device)
        
#             # print("x, y, y_hat shape", x.shape, y.shape, y_hat.shape)
#             loss = criterion(y_hat, y)

#             train_loss += loss.detach().cpu().item() / len(train_loader)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
#     y_hat = visualization(y_hat, 32, 3)
#         # y = visualization(y_slicing, 32, 3)
#     save_image(y_hat.cpu(), f"train_output.jpg")
#     # Test loop
#     with torch.no_grad():
#         correct, total = 0, 0
#         test_loss = 0.0
#         for batch in tqdm(test_loader, desc="Testing"):
#             x, y = batch
#             x, y = x.to(device), y.to(device)
#             print(x.shape, y.shape)
#             y_hat = model(x, y)

#             y_slicing = slicing(y, 5, 3).to(device)

#             loss = criterion(y_hat, y_slicing)
#             test_loss += loss.detach().cpu().item() / len(test_loader)

#             # correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
#             # total += len(x)
#         y_hat = visualization(y_hat, 32, 3)
#         # y = visualization(y_slicing, 32, 3)
#         save_image(y_hat.cpu(), f"test_output.jpg")
#         print(f"Test loss: {test_loss:.2f}")
#         # print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':
    # main()
    # imgs = torch.randint(0, 1, (2, 1, 64, 60))
    # print(imgs.shape)
    # patches = slicing(imgs, 20, 3)
    # print(patches.shape)
    symbols = ["601857", "600028"]

    years = range(2008, 2022)
    imgs = []
    labels = []
    for s in symbols:
        for y in years:
            f = open(f"./kchart/{s}_{y}_imgs.pk", "rb")
            imgs.extend((pk.load(f)))
            f = open(f"./kchart/{s}_{y}_labels.pk", "rb")
            labels.extend(pk.load(f))
    
    convert_tensor = ToTensor()
    # print(convert_tensor(imgs[0]).size())
    random.shuffle(imgs)
    random.shuffle(labels)
    imgs = [convert_tensor(img) for img in imgs]
    labels = [convert_tensor(img) for img in labels]
    data = list(zip(imgs, labels))
    # print(imgs[0].shape, labels[0].shape)
    idx_train = round(0.8*(len(data)))
    train_set = data[:idx_train]
    test_set = data[idx_train:]
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=64)

    # Defining model and training options
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViTAutoEncoder((1, 64, 60), (1, 32, 15), n_src_patches=20, n_tgt_patches=5, n_blocks=8, hidden_d=8, n_heads=2, out_d=10).to(device)
    model.generate_mask((128, 1, 32, 15))
    N_EPOCHS = 10
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x, y)

            # y = slicing(y, 5, 3).to(device)
        
            # print("x, y, y_hat shape", x.shape, y.shape, y_hat.shape)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
    # y_hat = visualization(y_hat, 32, 3)
        # y = visualization(y_slicing, 32, 3)
    save_image(y_hat.cpu(), f"train_output.jpg")
    # Test loop
    model.generate_mask((64, 1, 32, 15))
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            # print(x.shape, y.shape)
            y_hat = model(x, y)

            # y_slicing = slicing(y, 5, 3).to(device)

            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            # correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            # total += len(x)
        # y_hat = visualization(y_hat, 32, 3)
        # y = visualization(y_slicing, 32, 3)
        save_image(y_hat.cpu(), f"test_output.jpg")
        print(f"Test loss: {test_loss:.2f}")
        # print(f"Test accuracy: {correct / total * 100:.2f}%")
