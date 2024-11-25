import torch
from torchvision import transforms, datasets
from typing import Tuple, Dict
from torch.utils.data import Dataset, random_split

class ImageClassificationDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return {
            'pixel_values': image,
            'labels': label
        }

class DataModule:
    """数据模块类"""
    def __init__(self, config):
        self.config = config
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
    
    def _get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self):
        """准备数据集"""
        # 加载完整数据集
        full_dataset = datasets.ImageFolder(self.config.data_dir)
        
        # 计算划分大小
        total_size = len(full_dataset)
        train_size = int(self.config.train_ratio * total_size)
        val_size = int(self.config.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # 划分数据集
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 包装数据集
        self.train_dataset = ImageClassificationDataset(train_dataset, self.train_transform)
        self.val_dataset = ImageClassificationDataset(val_dataset, self.test_transform)
        self.test_dataset = ImageClassificationDataset(test_dataset, self.test_transform)
        
        # 保存数据集信息
        self.classes = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_info(self):
        """获取数据集信息"""
        return {
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset)
        }