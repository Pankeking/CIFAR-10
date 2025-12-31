from core.numpy_model import NumpyModel
from core.torch_model import TorchModel
from core.hc_model import HCModel
from nn.losses import Loss, LossMode
from nn.optimizers import Optimizer, OptimizerMode
import torch.optim as optim

def build_model(backend: str, optimizer: Optimizer, loss_mode: LossMode):
    loss = Loss(loss_mode=loss_mode)
    if backend == "torch":
        return TorchModel()
    elif backend == "numpy":
        return NumpyModel(loss=loss, optimizer=optimizer)
    elif backend == "hc":
        return HCModel(loss=loss, optimizer=optimizer)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def build_torch_optimizer(model, optimizer_config):
    params = model.parameters()
    
    if optimizer_config.optimizer_mode == OptimizerMode.ADAM:
        optimizer = optim.Adam(
            params,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.epsilon
        )
    elif optimizer_config.optimizer_mode == OptimizerMode.SGD:
        optimizer = optim.SGD(
            params,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config.optimizer_mode}")
    
    return optimizer
