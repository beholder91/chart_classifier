import torch
import timm
from torch import nn
from transformers import PretrainedConfig

class TimmConfig(PretrainedConfig):
    """Timm模型配置类"""
    model_type = "timm"
    
    def __init__(
        self,
        model_name="efficientnet_b0",
        num_classes=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes

class TimmModelForImageClassification(nn.Module):
    """Timm模型包装类"""
    def __init__(self, config):
        super().__init__()
        self.config = TimmConfig(
            model_name=config.model_name,
            num_classes=config.num_classes
        )
        self.model = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=config.num_classes
        )

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs, labels)
            return {"loss": loss, "logits": outputs}
        
        return {"logits": outputs}