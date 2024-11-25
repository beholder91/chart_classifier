import os
import torch
from PIL import Image
from torchvision import transforms
from models.model import TimmModelForImageClassification
from config import config
import glob
from typing import Union, List, Tuple
import time
from safetensors.torch import load_file

class ChartClassifier:
    def __init__(self, model_path: str = "./final_model"):
        """
        初始化图表分类器
        Args:
            model_path: 模型路径，默认为 ./final_model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._get_transform()
        self.labels = ["charts", "non-charts"]
        
    def _load_model(self) -> TimmModelForImageClassification:
        """加载模型"""
        try:
            # 构建模型文件路径
            model_file = os.path.join(self.model_path, "model.safetensors")
            
            # 检查文件是否存在
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"模型文件未找到: {model_file}")
            
            # 创建模型实例
            model = TimmModelForImageClassification(config)
            
            # 从safetensors文件加载权重
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            print(f"模型已成功加载自: {model_file}")
            return model
            
        except Exception as e:
            raise Exception(f"加载模型时出错: {str(e)}")

    def _get_transform(self) -> transforms.Compose:
        """获取图像预处理转换"""
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理单张图像
        Args:
            image_path: 图像路径
        Returns:
            处理后的图像张量
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image.unsqueeze(0).to(self.device)
        except Exception as e:
            raise Exception(f"处理图像 {image_path} 时出错: {str(e)}")

    def predict_single(self, image_path: str) -> Tuple[str, float]:
        """
        预测单张图像
        Args:
            image_path: 图像路径
        Returns:
            (预测类别, 置信度)
        """
        try:
            image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                pred_label = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_label].item()
            
            return self.labels[pred_label], confidence
            
        except Exception as e:
            raise Exception(f"预测图像 {image_path} 时出错: {str(e)}")

    def predict_batch(self, image_dir: str, batch_size: int = 32) -> List[dict]:
        """
        批量预测目录中的图像
        Args:
            image_dir: 图像目录
            batch_size: 批次大小
        Returns:
            预测结果列表
        """
        results = []
        image_paths = glob.glob(os.path.join(image_dir, "*.[jp][pn][g]"))  # 支持 jpg, jpeg, png
        
        if not image_paths:
            raise Exception(f"在 {image_dir} 中未找到图像文件")
        
        print(f"找到 {len(image_paths)} 个图像文件")
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                label, confidence = self.predict_single(image_path)
                results.append({
                    "image_path": image_path,
                    "predicted_label": label,
                    "confidence": confidence
                })
                
                if i % 10 == 0:
                    print(f"已处理 {i}/{len(image_paths)} 个图像...")
                    
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n处理完成！")
        print(f"总用时: {total_time:.2f} 秒")
        print(f"平均每张图像用时: {total_time/len(image_paths):.2f} 秒")
        
        return results

def main():
    # 使用示例
    try:
        # 初始化分类器
        classifier = ChartClassifier()
        
        # 单张图像预测
        image_path = "data/test_images/table.png"  # 替换为你的图像路径
        if os.path.exists(image_path):
            label, confidence = classifier.predict_single(image_path)
            print(f"\n单张图像预测结果:")
            print(f"图像: {image_path}")
            print(f"预测类别: {label}")
            print(f"置信度: {confidence:.4f}")
        
        # # 批量预测
        # image_dir = "path/to/your/image/directory"  # 替换为你的图像目录
        # if os.path.exists(image_dir):
        #     results = classifier.predict_batch(image_dir)
        #     print("\n批量预测结果:")
        #     for result in results:
        #         print(f"图像: {os.path.basename(result['image_path'])}")
        #         print(f"预测类别: {result['predicted_label']}")
        #         print(f"置信度: {result['confidence']:.4f}")
        #         print("-" * 50)
    
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()