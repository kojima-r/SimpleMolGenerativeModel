import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import time

device = 'cuda'

class SmiDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length=None):
        self.data_num = len(data)
        # データを１ずつずらす
        self.x = [d[:-1] for d in data]
        self.y = [d[1:] for d in data]
        self.max_length = max_length

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.x[idx]
        out_label =  self.y[idx]
        # LongTensor型に変換する
        out_x=[1]+out_data+[2]
        out_y=out_data+[2,0]
        out_x = torch.LongTensor(out_x)
        out_y = torch.LongTensor(out_y)
        return out_x, out_y
def collate_fn(batch):
    bx, by = list(zip(*batch))
    bx=torch.nn.utils.rnn.pad_sequence(bx, batch_first=True)
    by=torch.nn.utils.rnn.pad_sequence(by, batch_first=True)
    return bx, by


def load_smi_data(filename,vocab=None):
    data=[]
    if vocab is None:
        vocab={"<pad>":0,"<sos>":1,"<eos>":2}
    for smi in open(filename):
        v=[]
        for ch in smi:
            if ch not in vocab:
                vocab[ch]=len(vocab)
            i=vocab[ch]
            v.append(i)
        data.append(v)
    return data,vocab

class VAE(nn.Module):
    def __init__(self, z_dim,vocab):
        super(VAE, self).__init__()
        self.embeds = nn.Embedding(len(vocab), 512)
        self.lstm1 = nn.LSTM(512, 128)
        self.lstm2 = nn.LSTM(128, 64)
        self.dense_enc1 = nn.Linear(64, 64)
        self.dense_encmean = nn.Linear(64, z_dim)
        self.dense_encvar = nn.Linear(64, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 128)
        self.dense_dec2 = nn.Linear(128, 256)
        self.dense_dec3 = nn.Linear(256, len(vocab))
        # 損失関数はNLLLoss()を使う。LogSoftmaxを使う時はこれを使うらしい。
        self.loss_function = nn.NLLLoss()
        self.z_dim=z_dim
    
    def _encoder(self, x):
        x = F.relu(self.embeds(x))
        x,_=self.lstm1(x)
        x = F.relu(x)
        x,_=self.lstm2(x)
        x = F.relu(x)
        x = F.relu(self.dense_enc1(x))
        mean = self.dense_encmean(x)
        var = F.softplus(self.dense_encvar(x))
        return mean, var
    
    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(device)
        return mean + torch.sqrt(var) * epsilon
 
    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = torch.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z

    def loss(self, x):
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
        z = self._sample_z(mean, var)
        out = self._decoder(z)

        recons_loss = self.loss_function(out.view(-1,out.shape[2]), y.view(-1))
        recons_loss = torch.mean(recons_loss)
        lower_bound = [KL, recons_loss] 
        return sum(lower_bound)


    def generate(self,n,m):
        z = torch.randn((n,m,self.z_dim)).to(device)
        out = self._decoder(z)
        return out

#train_data,vocab=load_smi_data("train.smi")
train_data,vocab=load_smi_data("train.smi")
valid_data,vocab=load_smi_data("valid.smi",vocab=vocab)
print("#train_data:",len(train_data))
print("#valid_data:",len(valid_data))
print("#vocab",len(vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=256
trainset=SmiDataset(train_data)
trainloader = torch.utils.data.DataLoader(trainset, collate_fn=collate_fn,batch_size=batch_size)
model = VAE(64,vocab)
# 最適化の手法はSGDで。lossの減りに時間かかるけど、一旦はこれを使う。
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    all_loss=0
    for x,y in trainloader:
        model.zero_grad()
        #out,z = model(x)
        loss = model.loss(x)
        loss.backward()
        optimizer.step()
        loss=loss.detach().clone().numpy()
        all_loss = loss+all_loss
    print(all_loss)

model_path = 'model.pth'
torch.save(model.state_dict(), model_path)

o=model.generate(10,43)
out=o.detach().clone().numpy()
out_index=np.argmax(out,axis=2)

vocab_inv={v:k for k,v in vocab.items()}
for v in out_index:
    smi=[]
    for el in v:
        a=vocab_inv[int(el)]
        smi.append(a)
    smi_s="".join(smi)
    print(smi_s)

