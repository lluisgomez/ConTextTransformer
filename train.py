import time,os,json
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset

from einops import rearrange

import fasttext
import fasttext.util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)


class ConTextTransformer(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        modules=list(resnet50.children())[:-2]
        self.resnet50=nn.Sequential(*modules)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.num_cnn_features = 64  # 8x8
        self.dim_cnn_features = 2048
        self.dim_fasttext_features = 300

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_cnn_features + 1, dim))
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim)
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        encoder_norm = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, txt, mask=None):
        x = self.resnet50(img)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.cnn_feature_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x2 = self.fasttext_feature_to_embedding(txt.float())
        x = torch.cat((x,x2), dim=1)

        #tmp_mask = torch.zeros((img.shape[0], 1+self.num_cnn_features), dtype=torch.bool)
        #mask = torch.cat((tmp_mask.to(device), mask), dim=1)
        #x = self.transformer(x, src_key_padding_mask=mask)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


class ConTextDataset(Dataset):
    def __init__(self, json_file, root_dir, root_dir_txt, train=True, transform=None):
        with open(json_file) as f:
            data = json.load(f)
        self.train = train
        self.root_dir = root_dir
        self.root_dir_txt = root_dir_txt
        self.transform = transform
        if (self.train):
            self.samples = data['train']
        else:
            self.samples = data['test']

        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.fasttext = fasttext.load_model('cc.en.300.bin')
        self.dim_fasttext = self.fasttext.get_dimension()
        self.max_num_words = 64


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.samples[idx][0]+'.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text = np.zeros((self.max_num_words, self.dim_fasttext))
        text_mask = np.ones((self.max_num_words,), dtype=bool)
        text_name = os.path.join(self.root_dir_txt, self.samples[idx][0]+'.json')
        with open(text_name) as f:
            data = json.load(f)

        words = []
        if 'textAnnotations' in data.keys():
            for i in range(1,len(data['textAnnotations'])):
                word = data['textAnnotations'][i]['description']
                if len(word) > 2: words.append(word)

        words = list(set(words))
        for i,w in enumerate(words):
            if i>=self.max_num_words: break
            text[i,:] = self.fasttext.get_word_vector(w)
            text_mask[i] = False
        
        target = self.samples[idx][1] - 1

        return image, text, text_mask, target

json_file = '/datatmp/datasets/ConText/annotations/split_0.json'
img_dir = "/datatmp/datasets/ConText/data/JPEGImages/"
txt_dir = "/datatmp/datasets/ConText/ocr_labels/"
input_size = 256
data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_set = ConTextDataset(json_file, img_dir, txt_dir, True, data_transforms_train)
test_set  = ConTextDataset(json_file, img_dir, txt_dir, False, data_transforms_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data_img, data_txt, txt_mask, target) in enumerate(data_loader):
        data_img = data_img.to(device)
        data_txt = data_txt.to(device)
        txt_mask = txt_mask.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data_img, data_txt, txt_mask), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data_img)) + '/' + '{:5}'.format(total_samples) +
                 ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data_img, data_txt, txt_mask, target in data_loader:
            data_img = data_img.to(device)
            data_txt = data_txt.to(device)
            txt_mask = txt_mask.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data_img, data_txt, txt_mask), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    return correct_samples / total_samples

N_EPOCHS = 50
start_time = time.time()

model = ConTextTransformer(image_size=input_size, num_classes=28, channels=3, dim=256, depth=2, heads=4, mlp_dim=512)
model.to(device)
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
optimizer = torch.optim.Adam(params_to_update, lr=0.0001)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)

train_loss_history, test_loss_history = [], []
best_acc = 0.

for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_epoch(model, optimizer, train_loader, train_loss_history)
    acc = evaluate(model, test_loader, test_loss_history)
    if acc>best_acc: torch.save(model.state_dict(), 'all_best.pth')
    scheduler.step()

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
