import os
import random
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

random.seed(1)

class COVIDDataset(Dataset):
    def __init__(self, data_dir, fold=0, original_split=False, train=True, transform=None):
        """
        肺炎分类任务的Dataset，使用图像作为样本单位
            param data_dir: str, 数据集所在路径
            param transform: torch.transform，数据预处理方法
        """
        self.label_name = {"covid": 0, "pneumonia": 1, "regular": 2}
        self.fold = fold  # 用第几个fold来做验证
        self.original_split = original_split
        self.train = train
        self.data_info = self.get_img_info(data_dir)
        
        self.transform = transform
        print(f'Train = {train}; Fold = {fold}; Original split = {original_split}; Sample size = {len(self.data_info)};')
    
    def __getitem__(self, index):
        img_path = self.data_info[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        label = self.label_name[img_path.split('\\')[-2]]
        
        return img, label

    def __len__(self): # 查看样本的数量
        return len(self.data_info)
    
    def get_img_info(self, data_dir):
        # we use the dict data_info to save the meta data
        # {'split0':{"sample_name": [img_dirs, ...]}}
        data_info = {'split0':{}, 'split1':{}, 'split2':{}, 'split3':{}, 'split4':{}}
        for dirName,subdirList,fileList in os.walk(data_dir):
            for filename in fileList:  # 检查目录中的所有图片
                if ".png" in filename.lower() or ".jpg" in filename.lower():
                    split_ = dirName.split('\\')[1]
                    if 'frame' in filename:
                        sample_name = filename[:filename.index('frame')-1]
                    else:
                        sample_name = filename[:-4]  # 去掉.jpg/.png
                    if sample_name not in data_info[split_].keys():
                        data_info[split_][sample_name] = []
                    data_info[split_][sample_name].append(os.path.join(dirName,filename))

        # 将dict转化为list并返回
        data_info_list = []
        if self.original_split:
            if self.train:
                del data_info['split'+str(self.fold)]  # 训练集删去一个fold
                for split_ in data_info.keys():
                    for sample_name in data_info[split_].keys(): # 一个sample是一个视频
                        data_info_list.append(data_info[split_][sample_name])
            else:
                for sample_name in data_info['split'+str(self.fold)].keys():  # 验证集只用删去那个fold
                    data_info_list.append(data_info['split'+str(self.fold)][sample_name])
        else:
            # 将所有的视频（一个视频存在一个列表中）放入data_info_list
            for split_ in data_info.keys():
                for sample_name in data_info[split_].keys(): # 一个sample是一个视频
                    data_info_list.append(data_info[split_][sample_name])
            random.seed(1)
            random.shuffle(data_info_list)
            i = self.fold
            if self.train:
                data_info_list = data_info_list[:int(0.2*i*len(data_info_list))] + data_info_list[int(0.2*(i+1)*len(data_info_list)):]
            else:
                data_info_list = data_info_list[int(0.2*i*len(data_info_list)):int(0.2*(i+1)*len(data_info_list))]
            
        data_info = [] 
        for data_list in data_info_list: # 把所有视频对应的List拼在一起，变成图像数据集
            data_info = data_info + data_list
        
        return data_info


if __name__ == '__main__':
    data_dir = 'D:/数据集/POCUS_5fold'
    BATCH_SIZE = 128

    # 测试
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])
    train_data = COVIDDataset(data_dir=data_dir, fold=0, original_split=True, train=True, transform=transform)
    valid_data = COVIDDataset(data_dir=data_dir, fold=0, original_split=True, train=False, transform=transform)
    train_data = COVIDDataset(data_dir=data_dir, fold=1, original_split=True, train=True, transform=transform)
    valid_data = COVIDDataset(data_dir=data_dir, fold=1, original_split=True, train=False, transform=transform)
    train_data = COVIDDataset(data_dir=data_dir, fold=2, original_split=True, train=True, transform=transform)
    valid_data = COVIDDataset(data_dir=data_dir, fold=2, original_split=True, train=False, transform=transform)
    train_data = COVIDDataset(data_dir=data_dir, fold=3, original_split=True, train=True, transform=transform)
    valid_data = COVIDDataset(data_dir=data_dir, fold=3, original_split=True, train=False, transform=transform)
    train_data = COVIDDataset(data_dir=data_dir, fold=4, original_split=True, train=True, transform=transform)
    valid_data = COVIDDataset(data_dir=data_dir, fold=4, original_split=True, train=False, transform=transform)
    
    # 构建DataLoder，使用实例化后的数据集作为dataset
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    # for i, (frames, label) in enumerate(train_loader):
    #     print('Train:', frames.shape, frames.min(), frames.mean(), frames.max())
    # for i, (frames, label) in enumerate(valid_loader):
    #     print('Valid:', frames.shape, frames.min(), frames.mean(), frames.max())
    
        
        