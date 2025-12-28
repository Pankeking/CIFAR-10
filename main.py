import argparse
from model import Model

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--train", action="store_true")
    args.add_argument("--evaluate", action="store_true")
    args = args.parse_args()
    learning_rate = 1e-2
    weight_decay = 1e-3
    number_samples = 50000
    epochs = 60
    learning_rate_str = f"{learning_rate:.1e}".replace("e", "E")

    if args.train and args.evaluate:
        raise ValueError("Cannot train and evaluate at the same time")
    elif args.train:
        model = Model(learning_rate=learning_rate, weight_decay=weight_decay, loss_mode="cross_entropy", activation_function="relu")
        model.create_model(strategy="cifar10", number_samples=number_samples)
        model.train(strategy="layered", epochs=epochs, batch_size=512, metrics=True)
        model.save(f"fixed_model_{learning_rate_str}_{number_samples}.pkl")
    elif args.evaluate:
        model = Model(learning_rate=learning_rate, weight_decay=weight_decay, loss_mode="cross_entropy", activation_function="relu")
        model.load(f"fixed_model_{learning_rate_str}_{number_samples}.pkl")
        test_accuracy = model.evaluate()
        train_accuracy = model.evaluate_on_train()
    else:
        raise ValueError("No action specified")


if __name__ == "__main__":
    main()
