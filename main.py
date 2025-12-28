import argparse
from model import Model
from view import run_view
from utils.losses import Loss, LossMode
from utils.optimizers import Optimizer, OptimizerMode

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--train", action="store_true")
    args.add_argument("--evaluate", action="store_true")
    args.add_argument("--predict", action="store_true")
    args.add_argument("--start", type=int, default=0, help="Start index in test set")

    args = args.parse_args()
    learning_rate = 1e-2
    weight_decay = 3e-3
    number_samples = 50000
    epochs = 40
    hidden_layer_size = 512
    loss_mode = LossMode.CROSS_ENTROPY
    optimizer_mode = OptimizerMode.SGD
    model_filename = f"fixed_model_{learning_rate:.1e}_{loss_mode.value}_{number_samples}_{optimizer_mode.value}_{hidden_layer_size}_{epochs}.pkl"
    optimizer = Optimizer(
            optimizer_mode=optimizer_mode,
            start_epoch_decay=30,
            decay_rate=0.99
        )

    if sum([args.train, args.evaluate, args.predict]) > 1:
        raise ValueError("Use only one of --train, --evaluate or --predict")
    elif args.train:
        model = Model(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            activation_function="relu",
            optimizer=optimizer,
            loss=Loss(loss_mode=loss_mode),
        )
        model.create_model(
            number_samples=number_samples,
            hidden_layer_size=hidden_layer_size
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
