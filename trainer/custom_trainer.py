import os
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
import time

class CustomTrainer:
    """自定义训练器类"""
    def __init__(self, config, model, train_dataset, val_dataset, test_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        os.makedirs(self.config.final_model_dir, exist_ok=True)
        
        self.training_args = self._get_training_args()
        self.trainer = self._create_trainer()
        self.start_time = time.time()
    
    def _get_training_args(self):
        """获取训练参数"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            
            # 评估策略
            eval_steps=self.config.eval_steps,          # 使用配置的评估步数
            evaluation_strategy="steps",                # 按步数评估
            save_strategy="steps",                      # 按步数保存
            save_steps=self.config.save_steps,          # 使用配置的保存步数
            
            # 模型选择和保存
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=self.config.save_total_limit,
            greater_is_better=True,                     # F1 分数越大越好
            
            # 其他设置
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            local_rank=-1,
            ddp_backend=None,
            disable_tqdm=False,                         # 显示进度条
            
            # 额外设置
            dataloader_num_workers=4,                   # 数据加载器的工作进程数
            dataloader_pin_memory=True,                 # 使用 pin_memory 加速数据传输
            fp16=False,                                  # 使用半精度训练
            gradient_accumulation_steps=1,              # 梯度累积步数
        )
    
    def _create_trainer(self):
        """创建训练器"""
        from utils.metrics import compute_metrics
        
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )
    
    def _print_training_info(self):
        """打印训练配置信息"""
        print("\n训练配置:")
        print(f"模型: {self.config.model_name}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"训练轮数: {self.config.num_epochs}")
        print(f"权重衰减: {self.config.weight_decay}")
        print(f"预热比例: {self.config.warmup_ratio}")
        print(f"\n保存设置:")
        print(f"检查点保存目录: {self.config.output_dir}")
        print(f"每 {self.config.save_steps} 步保存一次检查点")
        print(f"保留最近的 {self.config.save_total_limit} 个检查点")
        print(f"\n早停设置:")
        print(f"耐心值: {self.config.early_stopping_patience} 次评估")
        print(f"阈值: {self.config.early_stopping_threshold}")
        print("\n" + "="*50)
    
    def _format_time(self, seconds):
        """格式化时间"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    def train(self):
        """训练模型"""
        try:
            self._print_training_info()
            print("\n开始训练...")
            
            # 训练模型
            train_result = self.trainer.train()
            
            # 计算训练时间
            train_time = time.time() - self.start_time
            
            # 打印训练结果
            print("\n" + "="*50)
            print("训练完成，结果摘要:")
            print(f"总训练步数: {train_result.global_step}")
            print(f"总训练损失: {train_result.training_loss:.4f}")
            print(f"训练用时: {self._format_time(train_time)}")
            print(f"平均每步用时: {(train_time/train_result.global_step):.2f}秒")
            
            # 保存最终模型
            final_save_path = self.config.final_model_dir
            self.trainer.save_model(final_save_path)
            print(f"\n最终模型已保存到: {final_save_path}")
            
            return train_result
        
        except Exception as e:
            print(f"\n训练过程中出现错误: {str(e)}")
            raise e
    
    def evaluate(self, test=False):
        """评估模型"""
        try:
            dataset = self.test_dataset if test else self.val_dataset
            dataset_name = "测试" if test else "验证"
            print(f"\n在{dataset_name}集上评估...")
            
            # 评估开始时间
            eval_start_time = time.time()
            
            # 进行评估
            results = self.trainer.evaluate(dataset)
            
            # 计算评估时间
            eval_time = time.time() - eval_start_time
            
            # 打印评估结果
            print(f"\n{dataset_name}集结果:")
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")
            print(f"评估用时: {self._format_time(eval_time)}")
            
            return results
            
        except Exception as e:
            print(f"\n评估过程中出现错误: {str(e)}")
            raise e