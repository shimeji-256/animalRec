import numpy as np
import os
import paddle
from paddle.io import Dataset, DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
# import paddle.fluid as fluid
from paddle.vision.datasets import DatasetFolder
import paddle.vision.transforms as transforms
from PIL import Image
import sys
import warnings
import bisect
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# 在训练中进行数据增强很重要。
train_tfm = transforms.Compose([
    # 将图像大小调整为固定形状 (height = width = 128)
    transforms.Resize((128, 128)),
    # ---------- TODO ----------
    # 在此处添加你的代码
    # transforms.RandomErasing(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),

    transforms.ToTensor(),
])


# 我们不需要在测试和验证中进行扩充。
# 只需要调整 PIL 图像的大小并将其转换为 Tensor。
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

import os

#此处代码为构造dataloader函数。
IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert("RGB")
    

def make_dataset(root, split, txtfile=None):
    images = []
    if txtfile is not None:
        ordertxt = open(txtfile)
        for line in ordertxt:
            data = line.strip()
            if is_image_file(data):
                imgpath = os.path.join(root,split,data)
            item = (imgpath, -1)
            images.append(item)
    else:
        root=os.path.join(root, split)
        for onelabel in os.listdir(root):
            newdir=os.path.join(root, onelabel)
            for img in os.listdir(newdir):
                imgpath=os.path.join(newdir,img)
                item = (imgpath, int(onelabel))
                images.append(item)
    return images

class ImageSet(Dataset):
    def __init__(self, root,split, transform=None, loader=default_loader,txtfile=None):
        self.samples = make_dataset(root,split,txtfile)
        self.root = root
        self.txtfile = txtfile
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, gt = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, gt

    def __len__(self):
        return len(self.samples)
    
# 更大的batch size通常会提供更稳定的梯度，但是GPU 显存是有限的，batch size过大，显存可能加载不了。
batch_size = 256
# 如果 out of memory,可以调小batch_size或者选用显存更大的GPU

# 构建数据集。数据集划分为训练集、验证集、公开测试集和私有测试集。私有测试集的标签在助教手中，助教将依据您对私有测试集的预测结果来打分。
# 代码main.ipynb要求一并上交，所以在实验报告中请如实作答。
    
data_root = 'work/data/animal/'
train_set = ImageSet(data_root,split="train", loader=lambda x: Image.open(x),transform=train_tfm)
val_set = ImageSet(data_root,split="val", loader=lambda x: Image.open(x), transform=test_tfm)
pub_test_set = ImageSet(data_root,split="pub_test", loader=lambda x: Image.open(x), transform=test_tfm)
pri_test_set = ImageSet(data_root,split="pri_test",txtfile="pri_test_only_name.txt",loader=lambda x: Image.open(x), transform=test_tfm)
semi_set = ImageSet(data_root,split="semi",txtfile="semi_only_name.txt",loader=lambda x: Image.open(x), transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
pub_test_loader = DataLoader(pub_test_set, batch_size=batch_size, shuffle=False, drop_last=False)
pri_test_loader = DataLoader(pri_test_set, batch_size=batch_size, shuffle=False, drop_last=False)

class CNN(nn.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        # 常用模块的参数：
        # paddle.nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        # paddle.nn.MaxPool2D(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2D(3, 64, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2D(2, 2, 0),

            nn.Conv2D(64, 128, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2D(2, 2, 0),

            nn.Conv2D(128, 256, 3, 1, 1),
            nn.GELU(),
            nn.MaxPool2D(2, 2, 0),

            # nn.Conv2D(256, 512, 3, 1, 1),
            # nn.BatchNorm2D(512),
            # nn.ReLU(),
            # nn.MaxPool2D(4, 4, 0),
        )
        self.res_layers = nn.Sequential(
            nn.Conv2D(256, 1024, 1, 1),
            # nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.MaxPool2D(2, 2, 0),
        )
        self.cnncon_layers = nn.Sequential(

            nn.Conv2D(256, 512, 3, 1, 1),
            nn.GELU(),
            

            nn.Conv2D(512, 1024, 3, 1, 1),
            # nn.BatchNorm2D(1024),
            nn.ReLU(),
            nn.MaxPool2D(2, 2, 0),
            # nn.MaxPool2D(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 1024),
            nn.GELU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 10]

        # 提取特征
        x = self.cnn_layers(x)
        tmp = self.res_layers(x)
        x = self.cnncon_layers(x)
        x = x + tmp
        x = x.flatten(1)
        # 特征通过全连接层转换得到最终的logits。
        x = self.fc_layers(x)
        return x
    # ---------- TODO ----------
    #在此处自由修改模型架构以进行进一步改进。如增加卷积层数等。
    #如果您想使用一些众所周知的架构，例如 ResNet50，请确保不要加载预训练的权重。另外，复杂的模型有过拟合的风

# 参数初始化配置
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)


from ViT import PatchEmbed, Block
class VisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_chans=3,
                 class_dim=10,
                 embed_dim=768,
                 depth=4,
                 num_heads=4,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **args):
        super().__init__()
        self.class_dim = class_dim

        self.num_features = self.embed_dim = embed_dim
        # 图片分块和降维，块大小为patch_size，最终块向量维度为768
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        # 分块数量
        num_patches = self.patch_embed.num_patches
        # 可学习的位置编码
        self.pos_embed = self.create_parameter(shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        # 人为追加class token，并使用该向量进行分类预测
        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        # transformer
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim,class_dim) if class_dim > 0 else Identity()

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)
    # 参数初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):

        B = paddle.shape(x)[0]
        # 将图片分块，并调整每个块向量的维度
        x = self.patch_embed(x)

        # 将class token与前面的分块进行拼接
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        # 将编码向量中加入位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 堆叠 transformer 结构
        for blk in self.blocks:
            x = blk(x)
        # LayerNorm
        x = self.norm(x)
        # 提取分类 tokens 的输出
        return x[:, 0]

    def forward(self, x):
        # 获取图像特征
        x = self.forward_features(x)
        # 图像分类
        x = self.head(x)
        return x
    
# 固定 random seed
def same_seeds(seed):
    paddle.seed(seed)
    np.random.seed(seed)

same_seeds(0)
do_semi = False
init_model=True

if init_model:
    # model=CNN()
    # model = VisionTransformer()
    model = paddle.vision.models.resnet18(pretrained=False,num_classes=10)

n_epochs = 50
learning_rate = 0.001
work_path = 'work/model'
# 损失函数cross-entropy
criterion = nn.CrossEntropyLoss()

# 初始化优化器，可以微调一些超参数，比如学习率。
grad_norm = paddle.nn.ClipGradByGlobalNorm(clip_norm=10)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=1e-3, grad_clip=grad_norm)

best_acc = 0.0
val_acc = 0.0
loss_record = {'train': {'loss': [], 'iter': []}, 'val': {'loss': [], 'iter': []}}   # for recording loss
acc_record = {'train': {'acc': [], 'iter': []}, 'val': {'acc': [], 'iter': []}}      # for recording accuracy
loss_iter = 0
acc_iter = 0

if do_semi:
    this_train_loader = semi_train_loader
    print("do_semi")
else:
    this_train_loader = train_loader


print("start train")
for epoch in range(n_epochs):
    model.train()
    train_num = 0.0
    train_loss = 0.0
    val_num = 0.0
    val_loss = 0.0
    accuracy_manager = paddle.metric.Accuracy()
    val_accuracy_manager = paddle.metric.Accuracy()

    for batch_id, data in enumerate(this_train_loader):
        x_data, y_data = data

        if y_data.dim()==1:
            y_data = paddle.unsqueeze(y_data, axis=1)
        logits = model(x_data)
        # print("43_y_data.shape",y_data.shape)
        loss = criterion(logits, y_data)
        acc = paddle.metric.accuracy(logits, y_data)
        accuracy_manager.update(acc)
        if batch_id % 10 == 0:
            loss_record['train']['loss'].append(loss.numpy())
            loss_record['train']['iter'].append(loss_iter)
            loss_iter += 1
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        train_loss += loss
        train_num += len(y_data)

    
    total_train_loss = (train_loss / train_num) * batch_size
    train_acc = accuracy_manager.accumulate()
    acc_record['train']['acc'].append(train_acc)
    acc_record['train']['iter'].append(acc_iter)
    acc_iter += 1
    print("#===epoch: {}, train loss is: {}, train acc is: {:2.2f}%===#".format(epoch, total_train_loss.numpy(), train_acc*100))

    # ---------- Validation ----------
    # 确保模型处于value模式，以便某些仅仅适用于训练的操作（如 dropout）被禁用并正常工作。
    model.eval()

    # 分批迭代验证集。
    for batch_id, data in enumerate(val_loader):
        # 一个batch由图像数据和相应的标签组成。
        x_data, y_data = data
        
        if y_data.dim()==1:
            y_data = paddle.unsqueeze(y_data, axis=1)
        with paddle.no_grad():
          logits = model(x_data)
        loss = criterion(logits, y_data)
        # 计算每个batch的 the accuracy 
        acc = paddle.metric.accuracy(logits, y_data)
        val_accuracy_manager.update(acc)
        # 记录 loss and accuracy.
        val_loss += loss
        val_num += len(y_data)

    
    total_val_loss = (val_loss / val_num) * batch_size
    loss_record['val']['loss'].append(total_val_loss.numpy())
    loss_record['val']['iter'].append(loss_iter)
    val_acc = val_accuracy_manager.accumulate()
    acc_record['val']['acc'].append(val_acc)
    acc_record['val']['iter'].append(acc_iter)
    print("#===epoch: {}, val loss is: {}, val acc is: {:2.2f}%===#".format(epoch, total_val_loss.numpy(), val_acc*100))
    # ===================save====================
    if val_acc > best_acc:
        best_acc = val_acc
        paddle.save(model.state_dict(), os.path.join(work_path, 'best_model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(work_path, 'best_optimizer.pdopt'))
        print(f"saved,best_val_acc={best_acc}")


paddle.save(model.state_dict(), os.path.join(work_path, 'final_model.pdparams'))
paddle.save(optimizer.state_dict(), os.path.join(work_path, 'final_optimizer.pdopt'))