import argparse

import torch

from core.factory import build_model, build_torch_optimizer
from core.hc_model import HCModel
from core.numpy_model import NumpyModel
from core.torch_model import TorchModel
from data.torch_datasets import get_cifar10_loaders, get_tiny_imagenet_loaders
from nn.losses import LossMode
from nn.optimizers import Optimizer, OptimizerMode
from ui.view import run_view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Run interactive viewer")
    parser.add_argument("--start", type=int, default=0, help="Start index in test set")
    parser.add_argument(
        "--dataset", type=str, default="tiny_imagenet", help="Dataset to use"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["numpy", "torch", "hc"],
        help="Backend to use: numpy or torch or hc",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device to use (auto-detects mps/cuda if not specified)",
    )

    args = parser.parse_args()
    dataset_name = args.dataset
    backend = args.backend

    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(args.device)

    loss_mode = LossMode.CROSS_ENTROPY
    optimizer_mode = OptimizerMode.ADAM

    if dataset_name == "cifar10":
        C_out = 32
        learning_rate = 1e-3
        weight_decay = 1e-4
        number_samples = 50000
        epochs = 10
        batch_size = 128
    elif dataset_name == "tiny_imagenet":
        C_out = 64
        learning_rate = 1e-3
        weight_decay = 1e-4
        number_samples = 100_000
        epochs = 100
        batch_size = 128
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    file_extension = ".pt" if backend == "torch" else ".pkl"

    model_filename = f"{dataset_name}_model_{loss_mode.value}_{number_samples}_{optimizer_mode.value}_{epochs}{file_extension}"

    optimizer_config = Optimizer(
        optimizer_mode=optimizer_mode,
        weight_decay=weight_decay,
        start_epoch_decay=60,
        decay_rate=0.97,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        learning_rate=learning_rate,
    )

    if sum([args.train, args.evaluate, args.predict]) > 1:
        raise ValueError("Use only one of --train, --evaluate or --predict")

    if args.train:
        if backend == "torch":
            # Torch-only path
            if dataset_name == "cifar10":
                train_loader, test_loader = get_cifar10_loaders(
                    "datasets", batch_size, device
                )
                num_classes = 10
            elif dataset_name == "tiny_imagenet":
                train_loader, test_loader = get_tiny_imagenet_loaders(
                    "datasets/tiny-imagenet-200",
                    batch_size,
                    device,
                    max_samples=number_samples,
                )
                base_ds = (
                    train_loader.dataset.dataset
                    if hasattr(train_loader.dataset, "dataset")
                    else train_loader.dataset
                )
                num_classes = len(base_ds.classes)
            else:
                raise ValueError(
                    f"Torch backend not implemented for dataset: {dataset_name}"
                )

            model = TorchModel(
                in_channels=3, num_classes=num_classes, base_channels=C_out
            )
            model.dataset_name = dataset_name
            model.to(device)
            model.train()

            first_batch, _ = next(iter(train_loader))
            _ = model(first_batch.to(device))

            num_params = sum(p.numel() for p in model.parameters())
            if num_params == 0:
                raise RuntimeError("Model has no trainable parameters!")

            torch_optimizer = build_torch_optimizer(model, optimizer_config)

            model.train_torch(
                epochs=epochs,
                train_loader=train_loader,
                optimizer=torch_optimizer,
                optimizer_config=optimizer_config,
                device=device,
                metrics=True,
            )

            model.save(model_filename)
            print("Evaluating on test set:")
            model.evaluate_torch(test_loader, device=device)

        else:
            model = build_model(backend, optimizer_config, loss_mode)
            model.create_model(
                number_samples=number_samples,
                dataset_name=dataset_name,
                C_out=C_out,
            )
            model.train(
                epochs=epochs,
                batch_size=batch_size,
                metrics=True,
            )
            model.save(model_filename)
            model.evaluate()
            model.evaluate_on_train()

    elif args.evaluate:
        if backend == "torch":
            if dataset_name == "cifar10":
                _, test_loader = get_cifar10_loaders("datasets", batch_size, device)
                num_classes = 10
            elif dataset_name == "tiny_imagenet":
                _, test_loader = get_tiny_imagenet_loaders(
                    "datasets/tiny-imagenet-200",
                    batch_size,
                    device,
                    max_samples=None,
                )
                base_ds = test_loader.dataset
                num_classes = len(base_ds.classes)
            else:
                raise ValueError(
                    f"Torch backend not implemented for dataset: {dataset_name}"
                )

            model = TorchModel(
                in_channels=3, num_classes=num_classes, base_channels=C_out
            )
            model.load(model_filename)
            model.to(device)
            model.evaluate_torch(test_loader, device=device)
        else:
            model = build_model(backend, optimizer_config, loss_mode)
            model.load(model_filename)
            model.evaluate()
            model.evaluate_on_train()

    elif args.predict:
        if backend == "numpy":
            model_cls = NumpyModel
        elif backend == "torch":
            model_cls = TorchModel
        elif backend == "hc":
            model_cls = HCModel
        else:
            raise ValueError(f"Unknown backend: {backend}")
        run_view(model_filename, model_cls=model_cls, start_index=args.start)

    else:
        raise ValueError("No action specified")


if __name__ == "__main__":
    main()
