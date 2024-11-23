from skimage import data, color, feature
import skimage.data
import cv2
import numpy as np
import sys


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy
from PIL import Image

import inspect
import time

start_time = time.time()
##### DATA TO DATASET #####
# ===================================================================
# 定義轉換（Transform），將 PIL.Image 轉換為 Tensor 並進行正規化
# transform_train = transforms.Compose([
#     transforms.CenterCrop(160),
#     #transforms.Resize((200, 200)),
#     transforms.RandomResizedCrop(160),
#     transforms.RandomRotation(degrees=10),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.CenterCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=3),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# datapath setting & producing dataset
data_path = r'D:\2310011_Liao\ML\emotion\dataset'
dataset = datasets.ImageFolder(root=data_path, transform=transform)
train_data, test_data, train_labels, test_labels = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')  # 保證圖片是 RGB 格式
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# 創建訓練集和測試集的 Dataset 實例
train_dataset = CustomDataset(data=train_data, transform=transform) 
test_dataset = CustomDataset(data=test_data, transform=transform)


batch_size = 25
# 使用 DataLoader 將資料集轉換為可批次處理的形式
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========== 檢視圖片 ===========
# for X, y in train_dataloader:
#     # 隨機選擇其中一張圖片
#     index = np.random.randint(0, batch_size)
#     sample_image = X[index].permute(1, 2, 0).numpy()  # 將 tensor 轉換為 NumPy array
#     #(1, 2, 0): 表示將原始的維度順序 (C, H, W) C 代表 channels(0)，H 代表 height(1)，W 代表 width(2)）
#     #重新排列成 (H, W, C) 的順序。

#     # 顯示原始圖片
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image, %s"%(y[index].item()))
#     plt.imshow(sample_image)
#     plt.axis('off')

#     # 顯示資料增強後的圖片
#     plt.subplot(1, 2, 2)
#     plt.title("Transformed Image, %s"%(y[index].item()))
#     transformed_image = transform(Image.fromarray((sample_image * 255).astype(np.uint8)))
#     plt.imshow(transformed_image.permute(1, 2, 0).numpy())
#     plt.axis('off')

#     plt.show()

#     break  # 只顯示一個 batch 中的第一張圖片
# sys.exit()
# ===================================================================
# 創建一個迭代器
# data_iter = iter(train_dataloader)
# # 獲取一個 batch 的資料
# features, labels = next(data_iter)


# testing DataLoader
for i, (X, y) in enumerate(train_dataloader):
    print('NUM:', i)
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
image_size_x = X.shape[2]
image_size_y = X.shape[3]


##### CNN model #####
# ===================================================================
class emotiona_CNN(nn.Module):
    def __init__(self):
        super(emotiona_CNN, self).__init__()

        #建立類神經網路各層
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        #### output_size = [(input_size + 2*padding - kernel_size)/stride] + 1 ####
        #### 經過MaxPool2d, 也適用以上公式計算 ####
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        #### output_size = [(input_size + 2*padding - kernel_size)/stride] + 1 ####
        #### 經過MaxPool2d, 也適用以上公式計算 ####
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        #### output_size = [(input_size + 2*padding - kernel_size)/stride] + 1 ####
        #### 經過MaxPool2d, kernel_size=2, stride=2，output size再減半 ####
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        #### output_size = [(input_size + 2*padding - kernel_size)/stride] + 1 ####
        #### 經過MaxPool2d, kernel_size=2, stride=2，output size再減半 ####
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        #### output_size = [(input_size + 2*padding - kernel_size)/stride] + 1 ####
        #### 經過MaxPool2d, kernel_size=2, stride=2，output size再減半 ####
        
        self.flatten = nn.Flatten()

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128*5*5, out_features=1600),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Linear(in_features=1600, out_features=800),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=400),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=6)
            
        )

    def forward(self, x):
        # 定義資料如何通過類神經網路各層
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        logits = self.layer5(x)
        return logits


# ===================================================================
# 若 CUDA 環境可用，則使用 GPU 計算，否則使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 建立類神經網路模型，並放置於 GPU 或 CPU 上
model = emotiona_CNN().to(device)
# Loss function 損失函數
loss_fn = nn.CrossEntropyLoss()
# Optimizer 學習優化器
LR = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
#optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ===================================================================
##### save the CNN #####
with open(r'D:\2310011_Liao\ML\emotion\emotion_CNN_source.txt', 'w') as file:
    file.write(inspect.getsource(emotiona_CNN))
file.close()

with open(r'D:\2310011_Liao\ML\emotion\emotion_CNN_source.txt', 'a') as file:
    file.write(f'\nOptimizer: {optimizer}')
file.close()

with open(r'D:\2310011_Liao\ML\emotion\emotion_CNN_source.txt', 'a') as file:
    file.write(f'\nLearning rate: {LR}')
file.close()

with open(r'D:\2310011_Liao\ML\emotion\emotion_CNN_source.txt', 'a') as file:
    file.write(f'\nBatch size: {batch_size}')
file.close()
    
with open(r'D:\2310011_Liao\ML\emotion\emotion_CNN_source.txt', 'a') as file:
    file.write(f'\nTransform_dataset: {transform}')
file.close()


##### TRAINING & TESTING function #####
# ===================================================================
def train_model(dataloader, model, loss_fn, optimizer):
    # 資料總筆數
    num_data = len(dataloader.dataset) #10000
    # 批次數量
    num_batches = len(dataloader) #10000/batch_size
    
    # 1. Dataloader 的長度： len(dataloader) 代表每個 epoch 中的 batch 數量。
    # 例如，如果你有 1000 個訓練樣本，batch 大小為 64，那麼每個 epoch 中，
    # len(dataloader) 會是 1000 / 64 = 15。這個值是根據你的資料集大小和 
    # batch 大小計算得到的。
    
    # 2. Dataloader.dataset 的長度： len(dataloader.dataset) 代表資料集中的樣
    # 本總數。繼續上面的例子，len(dataloader.dataset) 會是 1000。
    
    # 在訓練過程中，通常會使用 len(dataloader) 來確定每個 epoch 有多少個 batch。
    # 而在某些情況下，你可能需要知道整個資料集的大小，這時就可以使用 len(dataloader.dataset)。
    # 總結而言，這兩者的關係是，len(dataloader) * batch_size = len(dataloader.dataset)。

    # 將模型設定為訓練模式
    model.train()

    # 批次讀取資料進行訓練
    train_loss, train_correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 將資料放置於 GPU 或 CPU
        X, y = X.to(device), y.to(device)

        pred = model(X)         # 計算預測值
        loss = loss_fn(pred, y) # 計算損失值（loss）

        optimizer.zero_grad()   # 重設參數梯度（gradient）
        loss.backward()         # 反向傳播（backpropagation）
        optimizer.step()        # 更新參數
        
        # 計算每次batch預測loss的加總值
        train_loss += loss.item()
        # 計算每次batch預測正確數量的加總值
        train_correct += (pred.argmax(axis=1) == y).type(torch.float).sum().item()
        #print((pred.argmax(axis=1) == y).type(torch.float).sum().item())
        
        # 輸出訓練過程資訊
        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    # 計算平均損失值與平均正確率
    train_loss /= num_batches
    train_correct /= num_data
    print(f"Train Error: \n Accuracy: {(100*train_correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss, train_correct

# 測試模型
def test_model(dataloader, model, loss_fn):
    # 資料總筆數
    num_data = len(dataloader.dataset) #10000
    # 批次數量
    num_batches = len(dataloader) #10000/batch_size

    # 將模型設定為驗證模式
    model.eval()

    # 初始化數值
    test_loss, test_correct = 0, 0

    # 驗證模型準確度
    with torch.no_grad():  # 不要計算參數梯度
        for X, y in dataloader:
            # 將資料放置於 GPU 或 CPU
            X, y = X.to(device), y.to(device)

            # 計算預測值
            pred = model(X)

            # 計算損失值的加總值
            test_loss += loss_fn(pred, y).item()

            # 計算預測正確數量的加總值
            test_correct += (pred.argmax(axis=1) == y).type(torch.float).sum().item()

    # 計算平均損失值與平均正確率
    test_loss /= num_batches
    test_correct /= num_data
    print(f"Test Error: \n Accuracy: {(100*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, test_correct
   
    

##### START TRAINING the model ####
# ===================================================================  
epochs = 50
train_loss__ = []
train_correct__ = []
test_loss__ = []
test_correct__ = []
for t in range(epochs):
    se = time.time()
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_correct = train_model(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_correct = test_model(test_dataloader, model, loss_fn)
    print('OK, one time')
    ee = time.time()
    print('time', (ee-se)/60, 'min')
    
    train_loss__.append(train_loss)
    train_correct__.append(train_correct)
    
    test_loss__.append(test_loss)
    test_correct__.append(test_correct)
    print(' ')
print("完成！")


##### PLOTTING the loss & accuracy time series #####
# =================================================================== 
fig,ax = plt.subplots(figsize=(6,5))
plt.plot(np.arange(1, epochs+1), train_loss__, 'b')
plt.plot(np.arange(1, epochs+1), test_loss__, 'r')
plt.legend(['training loss', 'testing loss'])
plt.xlim(1, epochs)
plt.ylim(0, 2)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(linestyle='dashed', linewidth=0.5)
plt.title('LOSS')

fig,ax = plt.subplots(figsize=(6,5))
plt.plot(np.arange(1, epochs+1), np.array(train_correct__)*100, 'b')
plt.plot(np.arange(1, epochs+1), np.array(test_correct__)*100, 'r')
plt.legend(['training accuracy', 'testing accuracy'])
plt.xlim(1, epochs)
plt.ylim(10, 100)
plt.xlabel('epochs')
plt.ylabel('Accuracy [%]')
plt.grid(linestyle='dashed', linewidth=0.5)
plt.title('ACCURACY')
plt.show



##### SAVING the model parameters #####
# =================================================================== 
torch.save(model.state_dict(), "emotion_Convolution.pth")


##### TRYING the model #####
# =================================================================== 
# building the model
model2 = emotiona_CNN()
# loading the parameters
model2.load_state_dict(torch.load("emotion_Convolution.pth"))
# testing mode
model2.eval()

# 各類別名稱
classes = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 取得測試資料
randnum1 = np.random.randint(1,100, 1)[0]
x, y = test_dataset[randnum1][0].unsqueeze(0), test_dataset[randnum1][1]

with torch.no_grad(): # 不要計算參數梯度
    # 以模型進行預測
    pred = model2(x)

    # 整理測試結果
    predicted, actual = classes[pred[0].argmax(axis=0)], classes[y]
    print(f'預測值："{predicted}" / 實際值："{actual}"')
    
end_time = time.time()
print('COST TIME:', (end_time-start_time)/60, 'min')