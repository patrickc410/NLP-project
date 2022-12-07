from nlp_proj.model_bilstm_baseline import BiLSTMClassifier, BiLSTMRegressor
from nlp_proj.model_bilstm_multitask import BiLSTMMultitask
from typing import Tuple, Union, List, Dict
import torch.nn as nn
from types import SimpleNamespace
import torch.optim as optim

def make_model(config: SimpleNamespace) -> Tuple[nn.Module]:
    """Make model according to configuration"""

    model_arch = None
    model_params = None
    if config.architecture == "BiLSTMClassifier":
        model_arch = BiLSTMClassifier
        model_params = dict(
            num_classes=config.num_classes,
            vocab_size=config.vocab_size,
            dim_emb=config.dim_emb,
            dim_hid=config.dim_hid,
            num_layers=config.num_layers,
        )
    elif config.architecture == "BiLSTMRegressor":
        model_arch = BiLSTMRegressor
        model_params = dict(
            vocab_size=config.vocab_size,
            dim_emb=config.dim_emb,
            dim_hid=config.dim_hid,
            num_layers=config.num_layers,
        )
    elif config.architecture == "BiLSTMMultitask":
        model_arch = BiLSTMMultitask
        model_params = dict(
            num_classes_list=config.num_classes_list,
            vocab_size=config.vocab_size,
            dim_emb=config.dim_emb,
            dim_hid=config.dim_hid,
            num_layers=config.num_layers,
        )
    else:
        raise Exception(f"model architecture {config.architecture} not supported")
    
    model = model_arch(**model_params)
    return model



def make_optimizer(config: SimpleNamespace, model: nn.Module) -> optim.Optimizer:
    """ Make optimizer from configuration and model """

    optim_type = None
    optim_params = None 
    if config.optimizer.lower() == "adam":
        optim_type = optim.Adam
    if config.optimizer.lower() == "adamw":
        optim_type = optim.AdamW
    if config.optimizer.lower() == "radam":
        optim_type = optim.RAdam
    if config.optimizer.lower() == "sgd":
        optim_type = optim.SGD
    
    optim_params = dict(
        lr=config.base_lr,
        weight_decay=config.weight_decay,
    )

    
    optimizer = optim_type(model.parameters(), **optim_params)
    return optimizer
    

def _make_criterion(name: str, params: Dict = None) -> nn.Module:
    if params is None:
        params = dict()
    if name.lower() == "CrossEntropyLoss".lower():
        return nn.CrossEntropyLoss(**params)
    if name.lower() == "MSELoss".lower():
        return nn.MSELoss(**params)
    else:
        raise Exception(f"Criterion {name} not supported")
    

def make_criterion(config: SimpleNamespace) -> Union[nn.Module, List[nn.Module]]:
    """ Make criterion for single and multi task training """
    # Multi-task
    if config.multitask is True:
        criterion_list = []
        for crit_name in config.label_criterion:
            criterion = _make_criterion(crit_name)
            criterion_list.append(criterion)
        return criterion_list
    
    # Single-task
    criterion = _make_criterion(config.label_criterion)
    return criterion

    