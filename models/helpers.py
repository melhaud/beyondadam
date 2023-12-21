from typing import Dict

from torch.optim import Optimizer, Adam
from optimizers import OASIS
from models import FullyConnectedNetwork, CifarNet


def get_optimizer(optimizer_name: str,
                  model,
                  config: Dict) -> Optimizer:
    """Constructs an optimizer.

    Args:
        optimizer_name: The name of an optimizer.
        model: An instance of the model.
        config: The dictionary with optimizer hyperparameters.

    Returns:
        An instance of optimizer
    """
    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(),
                         lr=config.lr,
                         betas=(0.9, config.beta2))
    elif optimizer_name == "AMSGrad":
        optimizer = Adam(model.parameters(),
                         lr=config.lr,
                         betas=(0.9, config.beta2),
                         amsgrad=True)
    elif optimizer_name == "OASIS":
        optimizer = OASIS(model.parameters(),
                          lr=config.lr,
                          betas=(0.9, config.beta2))
    else:
        raise NotImplementedError("This optimizer is not supported!")
    return optimizer


def get_dataloaders(dataset_name: str,
                    batch_size: int,
                    **kwargs):
    if dataset_name == 'mnist':
      dataset = MNIST
      transform = transforms.Compose(
          [transforms.ToTensor(),
          transforms.Normalize((0.1307), (0.3081))])
    elif dataset_name == 'cifar10':
      dataset = CIFAR10
      transform = transforms.Compose(
          [transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822 ,0.4465),
                               (0.2470, 0.2435, 0.2616))])
    else:
      raise NotImplementedError("The dataset is not supported!")

    train_loader = DataLoader(
      dataset(
          './data', train=True, download=True, transform=transform
      ),
      batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = DataLoader(
      dataset(
          './data', train=False, transform=transform
      ),
      batch_size=batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader


def get_sweep_config(name: str) -> Dict:
    """Defines the config for grid search.

    Args:
        name: The name of a sweep/experiment.

    Returns:
        A sweep config.
    """
    sweep_config = {
        'method': 'random',
        'name': name
    }

    ######################
    metric = {
        'name': 'test_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    ######################
    parameters_dict = {
        'lr': {
            'values': [1e-2, 1e-3, 3e-4, 1e-4]
        },
        'beta2': {
            'values': [0.99, 0.999]
        }
    }
    sweep_config['parameters'] = parameters_dict

    return sweep_config

def get_model(model_name: str,
              input_dim) -> torch.nn.Module:
    """Constructs a model.

    Args:
        model_name: "FullyConnectedNetwork" or "CifarNet".
        input_dim: The input dimension.

    Returns:
        An instance of the model.
    """
    if model_name == "FullyConnectedNetwork":
        model = FullyConnectedNetwork(input_dim, 10).to(device)
    elif model_name == "CifarNet":
        model = CifarNet(32, 10).to(device)
    else:
        raise NotImplementedError("This model is not supported!")
    return model

def run_grid_search(dataset_name: str,
                    model_name: str,
                    optimizer_name: str,
                    config=None):
    with wandb.init(config=config):
        config = wandb.config
        if dataset_name == "mnist":
            input_dim = 1 * 28**2
        else:
            input_dim = 3 * 32**2
        model = get_model(model_name, input_dim)
        optimizer = get_optimizer(optimizer_name, model, config)
        train_loader, test_loader = get_dataloaders(dataset_name=dataset_name,
                                                    batch_size=BATCH_SIZE)
        baseline_experiment = Experimentation(config,
                                              train_loader,
                                              test_loader,
                                              model,
                                              optimizer=optimizer,
                                              device=device)
        baseline_experiment.experiment()