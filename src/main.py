import argparse
from core.model import Model
from ui.view import run_view
from nn.losses import Loss, LossMode
from nn.optimizers import Optimizer, OptimizerMode

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--train, -t", action="store_true", help="Train the model")
    args.add_argument("--evaluate, -e", action="store_true", help="Evaluate the model")
    
    args.add_argument("--predict, -p", action="store_true", help="Predict the model")
    args.add_argument("--start, -s", type=int, default=0, help="Start index in test set")

    args.add_argument("--dataset, -d", type=str, default="tiny_imagenet", help="Dataset to use")

    args = args.parse_args()
    learning_rate = 1e-3
    weight_decay = 1e-4
    number_samples = 5000
    epochs = 35
    loss_mode = LossMode.CROSS_ENTROPY
    optimizer_mode = OptimizerMode.ADAM
    dataset_name = args.dataset
    
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
        model = Model(
            optimizer=optimizer,
            loss=Loss(loss_mode=loss_mode),
        )
        model.create_model(
            number_samples=number_samples,
            dataset_name=dataset_name,
        )
        model.train(
            epochs=epochs,
            batch_size=512,
            metrics=True
        )
        model.save(model_filename)
        model.evaluate()

    elif args.evaluate:
        model = Model()
        model.load(model_filename)
        model.evaluate()
        model.evaluate_on_train()
    elif args.predict:
        run_view(model_filename, start_index=args.start)
    else:
        raise ValueError("No action specified")


if __name__ == "__main__":
    main()
