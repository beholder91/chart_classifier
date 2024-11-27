import os
import torch
from PIL import Image
from torchvision import transforms
from models.model import TimmModelForImageClassification
from config import config
import glob
from typing import Union, List, Tuple, Dict
import time
from safetensors.torch import load_file
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class InferenceDataset(Dataset):
    """推理数据集类"""
    def __init__(self, image_paths: List[Union[str, Path]], transform=None):
        self.image_paths = image_paths
        self.transform = transform if transform else self._get_default_transform()
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'pixel_values': image, 'image_path': str(image_path)}

class ChartClassifier:
    """图表分类器"""
    def __init__(self, model_path: str = "./final_model"):
        """
        初始化分类器
        Args:
            model_path: 模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.labels = ["charts", "non-charts"]  # 与训练保持一致
        
    def _load_model(self, model_path: str) -> TimmModelForImageClassification:
        """加载模型"""
        try:
            model_file = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"模型文件未找到: {model_file}")
            
            # 创建模型实例，保持与训练时相同的配置
            model = TimmModelForImageClassification(config)
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict)
            
            return model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"模型加载错误: {str(e)}")

    @torch.no_grad()  # 优化推理性能
    def predict_single(self, image_path: Union[str, Path]) -> Dict:
        """
        预测单张图像
        Args:
            image_path: 图像路径
        Returns:
            包含预测结果的字典
        """
        try:
            # 创建数据集和加载器
            dataset = InferenceDataset([image_path])
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # 获取预测结果
            batch = next(iter(loader))
            pixel_values = batch['pixel_values'].to(self.device)
            
            # 模型推理
            outputs = self.model(pixel_values)
            probabilities = torch.softmax(outputs['logits'], dim=1)
            pred_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_label].item()
            
            return {
                'image_path': str(image_path),
                'predicted_label': self.labels[pred_label],
                'confidence': confidence
            }
            
        except Exception as e:
            raise Exception(f"预测失败 {image_path}: {str(e)}")

    def predict_batch(self, image_dir: Union[str, Path], batch_size: int = 32) -> List[Dict]:
        """
        批量预测目录中的图像
        Args:
            image_dir: 图像目录路径
            batch_size: 批处理大小
        Returns:
            预测结果列表，并保存详细结果到文件
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"目录不存在: {image_dir}")
        
        # 获取所有支持的图像文件
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            raise ValueError(f"在 {image_dir} 中未找到图像文件")
        
        # 创建结果目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = Path("results") / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建数据集和加载器
        dataset = InferenceDataset(image_paths)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        results = []
        total_batches = len(loader)
        total_images = len(image_paths)
        start_time = time.time()
        
        # 统计信息
        stats = {label: 0 for label in self.labels}
        high_conf_count = 0  # 高置信度预测数
        low_conf_count = 0   # 低置信度预测数
        conf_threshold = 0.9 # 高置信度阈值
        
        print(f"\n开始处理 {total_images} 个图像...")
        print(f"批次大小: {batch_size}")
        print(f"总批次数: {total_batches}")
        print("=" * 50)
        
        # 打开日志文件
        log_file = result_dir / "prediction_details.txt"
        with log_file.open('w', encoding='utf-8') as f:
            f.write(f"预测开始时间: {timestamp}\n")
            f.write(f"总图像数: {total_images}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, batch in enumerate(loader):
                try:
                    batch_start_time = time.time()
                    pixel_values = batch['pixel_values'].to(self.device)
                    image_paths = batch['image_path']
                    
                    # 模型推理
                    outputs = self.model(pixel_values)
                    probabilities = torch.softmax(outputs['logits'], dim=1)
                    pred_labels = torch.argmax(probabilities, dim=1)
                    confidences = probabilities.gather(1, pred_labels.unsqueeze(1))
                    
                    # 处理批次结果
                    for path, label, conf in zip(image_paths, pred_labels, confidences):
                        label_name = self.labels[label.item()]
                        conf_value = conf.item()
                        
                        # 更新统计信息
                        stats[label_name] += 1
                        if conf_value >= conf_threshold:
                            high_conf_count += 1
                        else:
                            low_conf_count += 1
                        
                        result = {
                            'image_path': path,
                            'predicted_label': label_name,
                            'confidence': conf_value
                        }
                        results.append(result)
                        
                        # 写入详细日志
                        f.write(f"图像: {path}\n")
                        f.write(f"预测类别: {label_name}\n")
                        f.write(f"置信度: {conf_value:.4f}\n")
                        f.write("-" * 30 + "\n")
                    
                    # 批次处理完成后的日志
                    batch_time = time.time() - batch_start_time
                    images_processed = (i + 1) * batch_size
                    images_processed = min(images_processed, total_images)
                    progress = images_processed / total_images * 100
                    
                    print(f"批次 [{i+1}/{total_batches}] "
                        f"处理: {images_processed}/{total_images} ({progress:.1f}%) "
                        f"用时: {batch_time:.2f}秒")
                    
                except Exception as e:
                    print(f"处理批次 {i+1} 时出错: {str(e)}")
                    f.write(f"错误批次 {i+1}: {str(e)}\n")
                    continue
        
        # 计算总体统计信息
        process_time = time.time() - start_time
        avg_time_per_image = process_time / total_images
        
        # 生成并保存汇总报告
        summary_file = result_dir / "summary_report.txt"
        with summary_file.open('w', encoding='utf-8') as f:
            f.write("预测结果汇总\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总图像数: {total_images}\n")
            f.write(f"总用时: {process_time:.2f} 秒\n")
            f.write(f"平均每张用时: {avg_time_per_image:.4f} 秒\n\n")
            
            f.write("类别分布:\n")
            for label, count in stats.items():
                percentage = (count / total_images) * 100
                f.write(f"{label}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n高置信度预测 (>= {conf_threshold}): "
                    f"{high_conf_count} ({high_conf_count/total_images*100:.1f}%)\n")
            f.write(f"低置信度预测 (< {conf_threshold}): "
                    f"{low_conf_count} ({low_conf_count/total_images*100:.1f}%)\n")
        
        # 打印汇总信息
        print("\n" + "=" * 50)
        print("处理完成！")
        print(f"结果保存在: {result_dir}")
        print("\n类别分布:")
        for label, count in stats.items():
            print(f"{label}: {count} ({count/total_images*100:.1f}%)")
        print(f"\n高置信度预测 (>= {conf_threshold}): {high_conf_count} ({high_conf_count/total_images*100:.1f}%)")
        print(f"总用时: {process_time:.2f} 秒")
        print(f"平均每张用时: {avg_time_per_image:.4f} 秒")
        
        return results

def main():
    """主函数"""
    try:
        # 初始化分类器
        classifier = ChartClassifier()
        
        # 单张图像预测示例
        # image_path = "data/test_images/table.png"
        # if os.path.exists(image_path):
        #     result = classifier.predict_single(image_path)
        #     print("\n单张图像预测结果:")
        #     print(f"图像: {result['image_path']}")
        #     print(f"预测类别: {result['predicted_label']}")
        #     print(f"置信度: {result['confidence']:.4f}")
        
        # # 批量预测示例
        image_dir = "data/test_images"
        if os.path.exists(image_dir):
            results = classifier.predict_batch(image_dir)
        
    except Exception as e:
        print(f"运行错误: {str(e)}")

if __name__ == "__main__":
    main()