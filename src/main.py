import argparse
from core.model import NumpyModel
from ui.view import run_view
from nn.losses import Loss, LossMode
from nn.optimizers import Optimizer, OptimizerMode

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--train", action="store_true", help="Train the model")
    args.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    
    args.add_argument("--predict", action="store_true", help="Predict the model")
    args.add_argument("--start", type=int, default=0, help="Start index in test set")

    args.add_argument("--dataset", type=str, default="tiny_imagenet", help="Dataset to use")

    args = args.parse_args()
    dataset_name = args.dataset

    loss_mode = LossMode.CROSS_ENTROPY
    optimizer_mode = OptimizerMode.ADAM
    
    if dataset_name == "cifar10":
        C_out = 32
        learning_rate = 1e-3
        weight_decay = 1e-4
        number_samples = 50000
        epochs = 40
        batch_size = 128
    elif dataset_name == "tiny_imagenet":
        C_out = 64
        learning_rate = 1e-3
        weight_decay = 3e-4
        number_samples = 100000
        epochs = 50
        batch_size = 256
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    model_filename = f"{dataset_name}_model_{learning_rate:.1e}_{loss_mode.value}_{number_samples}_{optimizer_mode.value}_{epochs}_.pkl"
    optimizer = Optimizer(
            optimizer_mode=optimizer_mode,
            weight_decay=weight_decay,
            start_epoch_decay=22,
            decay_rate=0.98,
            beta1 = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8,
            learning_rate=learning_rate
        )

    if sum([args.train, args.evaluate, args.predict]) > 1:
        raise ValueError("Use only one of --train, --evaluate or --predict")
    elif args.train:
        model = NumpyModel(
            optimizer=optimizer,
            loss=Loss(loss_mode=loss_mode),
        )
        model.create_model(
            number_samples=number_samples,
            dataset_name=dataset_name,
            C_out=C_out,
        )
        model.train(
            epochs=epochs,
            batch_size=batch_size,
            metrics=True
        )
        model.save(model_filename)
        model.evaluate()
        model.evaluate_on_train()

    elif args.evaluate:
        model = NumpyModel()
        model.load(model_filename)
        model.evaluate()
        model.evaluate_on_train()
    elif args.predict:
        run_view(model_filename, start_index=args.start)
    else:
        raise ValueError("No action specified")


if __name__ == "__main__":
    main()
