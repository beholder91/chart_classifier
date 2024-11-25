from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TrainingConfig:
    # 数据相关
    data_dir: str = "data/dataset"
    img_size: int = 224
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 训练相关
    model_name: str = "efficientnet_b0"
    num_classes: int = 2
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # 早停和检查点相关
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0001
    save_total_limit: int = 7  # 大于 early_stopping_patience + 1
    save_steps: int = 100      # 每100步保存一次
    eval_steps: int = 100      # 每100步评估一次
    
    # 路径相关
    output_dir: str = "./results"
    logging_dir: str = "./logs"
    final_model_dir: str = "./final_model"
    
    # GPU相关
    gpu_id: int = 0
    
    # 日志相关
    logging_steps: int = 20

config = TrainingConfig()