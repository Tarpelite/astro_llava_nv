import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
import logging
import math

class NaNDetector:
    def __init__(self, model: nn.Module, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.hooks = []
        self.nan_detected = False
        self.last_input = None
        self.last_output = None
        
    def _hook_fn(self, module: nn.Module, input_: Tuple, output: Any, name: str) -> None:
        """检查每一层的输入和输出是否有NaN"""
        def _check_tensor(x, tensor_name, is_input=True):
            if isinstance(x, torch.Tensor):
                if torch.isnan(x).any():
                    self.nan_detected = True
                    type_str = "input" if is_input else "output"
                    print(f"NaN detected in {type_str} of {name}")
                    print(f"Tensor stats for {tensor_name}:")
                    print(f"  - Shape: {x.shape}")
                    print(f"  - Mean: {torch.mean(x[~torch.isnan(x)]) if torch.sum(~torch.isnan(x)) > 0 else 'all NaN'}")
                    print(f"  - Std: {torch.std(x[~torch.isnan(x)]) if torch.sum(~torch.isnan(x)) > 0 else 'all NaN'}")
                    print(f"  - Min: {torch.min(x[~torch.isnan(x)]) if torch.sum(~torch.isnan(x)) > 0 else 'all NaN'}")
                    print(f"  - Max: {torch.max(x[~torch.isnan(x)]) if torch.sum(~torch.isnan(x)) > 0 else 'all NaN'}")
                    print(f"  - NaN count: {torch.isnan(x).sum().item()}/{x.numel()}")
                    
                    if not is_input:
                        self.last_input = input_
                        self.last_output = output
        
        # 检查输入
        if isinstance(input_, tuple):
            for idx, inp in enumerate(input_):
                _check_tensor(inp, f"input_{idx}", True)
        else:
            _check_tensor(input_, "input", True)
            
        # 检查输出
        _check_tensor(output, "output", False)

    def register_hooks(self):
        """为模型的所有层注册钩子"""
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(
                lambda mod, inp, out, name=name: self._hook_fn(mod, inp, out, name)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def debug_forward_pass(model: nn.Module, sample_input: Dict[str, Any], logger=None):
    """执行一次前向传播并检查NaN"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # 开启调试模式
    torch.set_anomaly_enabled(True)
    
    # 创建NaN检测器
    detector = NaNDetector(model, logger)
    detector.register_hooks()
    
    try:
        # 执行前向传播
        with torch.autograd.detect_anomaly():
            outputs = model(**sample_input)
            
        if detector.nan_detected:
            logger.error("NaN detected during forward pass")
            if detector.last_input is not None:
                logger.error("Last layer input before NaN:")
                for idx, inp in enumerate(detector.last_input):
                    if isinstance(inp, torch.Tensor):
                        logger.error(f"Input {idx} stats:")
                        logger.error(f"  - Shape: {inp.shape}")
                        logger.error(f"  - Mean: {torch.mean(inp)}")
                        logger.error(f"  - Std: {torch.std(inp)}")
            
            return False, detector.last_input, detector.last_output
    except RuntimeError as e:
        logger.error(f"Runtime error during forward pass: {str(e)}")
        return False, None, None
    finally:
        detector.remove_hooks()
        torch.set_anomaly_enabled(False)
    
    return True, None, None

def check_gradients(model: nn.Module, logger=None):
    """检查模型梯度"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            if torch.isnan(grad).any():
                logger.error(f"NaN gradient detected in {name}")
                logger.error(f"Gradient stats:")
                logger.error(f"  - Shape: {grad.shape}")
                logger.error(f"  - NaN count: {torch.isnan(grad).sum().item()}/{grad.numel()}")
            elif torch.isinf(grad).any():
                logger.error(f"Inf gradient detected in {name}")
                logger.error(f"Gradient stats:")
                logger.error(f"  - Shape: {grad.shape}")
                logger.error(f"  - Inf count: {torch.isinf(grad).sum().item()}/{grad.numel()}")
            else:
                grad_abs = torch.abs(grad)
                if grad_abs.max() > 1000:
                    logger.warning(f"Large gradient detected in {name}")
                    logger.warning(f"Gradient stats:")
                    logger.warning(f"  - Max abs: {grad_abs.max().item()}")
                    logger.warning(f"  - Mean abs: {grad_abs.mean().item()}")

class GradientDebugger:
    def __init__(self, model: nn.Module, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.hooks = []
        
    def register_hooks(self):
        """注册梯度钩子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hooks.append((name, hook))
    
    def _gradient_hook(self, grad: torch.Tensor, name: str):
        """检查梯度"""
        if grad is None:
            return
        
        if torch.isnan(grad).any():
            print(f"NaN gradient in {name}")
            return None  # 返回None将阻止梯度传播
        
        if torch.isinf(grad).any():
            print(f"Inf gradient in {name}")
            return None
        
        grad_norm = torch.norm(grad)
        if grad_norm > 1000:
            self.logger.warning(f"Large gradient norm ({grad_norm:.2f}) in {name}")
            
        return grad
    
    def remove_hooks(self):
        """移除所有钩子"""
        for _, hook in self.hooks:
            hook.remove()
        self.hooks.clear()