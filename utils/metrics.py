from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    accuracy = (predictions == labels).mean()
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def generate_classification_report(predictions, labels, class_names):
    """生成分类报告"""
    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4
    )