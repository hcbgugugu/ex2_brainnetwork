#神经网络实验2

from torch.utils.data import Dataset, DataLoader
import os
from ex2_brainnetwork.ex2_brainnetwork.setting import BATCH_SIZE, SEED, CHAR_NUMBER
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device:{}'.format(device))
torch.cuda.empty_cache()

class ImageDataSet(Dataset):
    def __init__(self, dir_path):
        super(ImageDataSet, self).__init__()
        self.img_path_list = [f"{dir_path}/{filename}" for filename in os.listdir(dir_path)]
        self.trans = transforms.Compose([
            transforms.Grayscale(),# 每张图片都会被这行代码灰度化
            transforms.ToTensor(),
            #transforms.Resize((100, 40))
        ])

    def __getitem__(self, idx):
        image = self.trans(Image.open(self.img_path_list[idx]))
        #print(self.img_path_list[idx])
        label = self.img_path_list[idx].split("_")[0].split('/')[-1]
        label = one_hot_encode(label)
        return image, label

    def __len__(self):
        return len(self.img_path_list)

# 用torch.zeros()函数生成一个4行62列，值全是0的张量。接着循环标签中的各个字符，将字符在SEED中对应的索引获取到，然后将张量中对应位置的0，改成1。最后我们要返回一个一维的列表，长度是4*62=248
def one_hot_encode(label):
    """将字符转为独热码"""
    cols = len(SEED)
    rows = CHAR_NUMBER
    result = torch.zeros((rows, cols), dtype=float)
    for i, char in enumerate(label):
        j = SEED.index(char)
        result[i, j] = 1.0
    return result.view(1, -1)[0]


# 将模型预测的值从一维转成4行62列的二维张量，然后调用torch.argmax()函数寻找每一行最大值（也就是1）的索引。知道索引后就可以从SEED中找到对应的字符
def one_hot_decode(pred_result):
    """将独热码转为字符"""
    pred_result = pred_result.view(-1, len(SEED))#将 pred_result 重塑为一个二维张量，其中第一维的大小是自动计算的，第二维的大小等于 SEED 的长度。这通常用于确保 pred_result 是一个正确的二维独热编码张量，其中每行表示一个样本的独热编码，每列表示一个可能的类别。
    index_list = torch.argmax(pred_result, dim=1)
    text = "".join([SEED[i] for i in index_list])
    return text


def get_loader(path):
    """加载数据"""
    dataset = ImageDataSet(path)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    return dataloader
class NeuralNetWork(nn.Module):
    """CNN"""
    def __init__(self):
        super(NeuralNetWork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=30720, out_features=4096),
            nn.Dropout(0.2),#nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=CHAR_NUMBER * len(SEED))
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"损失值: {loss:>7f}")


def begintrain():
    model = NeuralNetWork().to(device)
    loss_fn = nn.MultiLabelSoftMarginLoss() #多标签分类损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #Adam优化方法
    train_dataloader = get_loader(f"./trainset")
    epoch = 30
    for t in range(epoch):
        print(f"训练周期 {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("\n")
    torch.save(model.state_dict(), f"./CHAPTCHAmodel.pth")
    print("训练完成，模型已保存")


def predict(model, file_path):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    with torch.no_grad():
        X = trans(Image.open(file_path)).reshape(1, 1, 60, 160)
        pred = model(X)
        text = one_hot_decode(pred)
        return text


def begintest():
    model = NeuralNetWork().to(device)
    model.load_state_dict(torch.load(f"./CHAPTCHAmodel.pth", map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    test_dir = f"./testset"
    total = len(os.listdir(test_dir))
    for filename in os.listdir(test_dir):
        file_path = f"{test_dir}/{filename}"
        real_captcha = file_path.split("-")[-1].replace(".png", "")
        pred_captcha = predict(model, file_path)
        if pred_captcha == real_captcha:
            correct += 1
            print(f"{file_path}的预测结果为{pred_captcha}，预测正确")
        else:
            print(f"{file_path}的预测结果为{pred_captcha}，预测错误")
    accuracy = f"{correct / total * 100:.2f}%"
    print(accuracy)

if __name__ == '__main__':
    # 训练集和测试集路径
    train_dataloader = get_loader(f"./trainset")
    test_dataloader = get_loader(f"./testset")
    for X, y in train_dataloader:
        print(X.shape)
        print(y.shape)
        break
    begintrain()
    begintest()