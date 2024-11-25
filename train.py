import os
import torch
from config.config import config
from data.dataset import DataModule
from models.model import TimmModelForImageClassification
from trainer.custom_trainer import CustomTrainer
from utils.metrics import generate_classification_report

def setup_environment():
    """设置环境"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
        print(f"当前设备号: {device}")
    else:
        print("使用CPU训练")

def main():
    # 设置环境
    setup_environment()
    
    try:
        # 准备数据
        data_module = DataModule(config)
        train_dataset, val_dataset, test_dataset = data_module.setup()
        
        # 打印数据集信息
        dataset_info = data_module.get_info()
        print("\n数据集信息:")
        for key, value in dataset_info.items():
            print(f"{key}: {value}")
        
        # 创建模型
        model = TimmModelForImageClassification(config)
        
        # 创建训练器
        trainer = CustomTrainer(
            config,
            model,
            train_dataset,
            val_dataset,
            test_dataset
        )
        
        # 训练模型
        train_result = trainer.train()
        
        # 在测试集上评估
        test_results = trainer.evaluate(test=True)
        
        print("\n训练完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main()