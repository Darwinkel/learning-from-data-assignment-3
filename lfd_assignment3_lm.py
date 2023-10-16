#!/usr/bin/env python3.11
"""Run the second part of assignment 3 with large language models."""

import argparse
import random as python_random

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

OPTIMISERS = {
    "sgd": SGD,
    "adam": Adam,
}

LOSS_FUNCTIONS = {"categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(from_logits=True)}


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--train_file",
        default="data/train.txt",
        type=str,
        help="Input file to learn from (default data/train.txt)",
    )
    parser.add_argument(
        "-d",
        "--dev_file",
        default="data/dev.txt",
        type=str,
        help="Separate dev set to read in (default data/dev.txt)",
    )
    parser.add_argument("-t", "--test_file", type=str, help="If added, use trained model to predict on test set")
    parser.add_argument("-lr", "--learning_rate", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("-ep", "--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("-p", "--patience", default=2, type=int, help="Amount of epochs before early stopping")
    parser.add_argument(
        "-opt",
        "--optimiser",
        default="adam",
        type=str,
        help="The optimiser we are using (default adam)",
    )
    parser.add_argument(
        "-lf",
        "--loss_function",
        default="categorical_crossentropy",
        type=str,
        help="The loss function we are using (default categorical_crossentropy)",
    )
    parser.add_argument(
        "-s",
        "--search",
        action="store_true",
        help="Perform full hyperparameter search (warning: takes a while!)",
    )
    parser.add_argument("-lm", "--language_model", default="distilbert-base-uncased", type=str, help="Language model")
    parser.add_argument("-msl", "-max_sequence_length", default=100, type=int, help="Max sequence length (default 100)")
    parser.add_argument("-lrs", "--learning_rate_scheduler", action="store_true", help="Use a learning rate scheduler")
    parser.add_argument(
        "-ilr",
        "--initial_learning_rate",
        default=1e-3,
        type=float,
        help="Initial learning rate (for scheduler)",
    )
    parser.add_argument("-ds", "--decay_steps", default=1000, type=int, help="Decay steps (for scheduler)")
    parser.add_argument(
        "-elr",
        "--end_learning_rate",
        default=1e-5,
        type=float,
        help="End learning rate (for scheduler)",
    )
    parser.add_argument(
        "-mt",
        "--max_trials",
        default=25,
        type=int,
        help="Maximum number of hyperparameter search trials",
    )
    return parser.parse_args()


def read_corpus(corpus_file):
    """Read in review data set and returns docs and labels."""
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def create_model(params):
    """Create the LM to use."""
    # Learning rate scheduler
    if params["use_learning_rate_scheduler"]:
        lrs = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["initial_learning_rate"],
            decay_steps=params["decay_steps"],
            decay_rate=params["decay_rate"],
        )
        optim = params["optimiser"](learning_rate=lrs)
    else:
        optim = params["optimiser"](learning_rate=params["learning_rate"])
    model = TFAutoModelForSequenceClassification.from_pretrained(
        params["language_model"],
        num_labels=params["num_labels"],
    )
    model.compile(loss=params["loss_function"], optimizer=optim, metrics=["accuracy"])
    return model


class CustomHyperModel(kt.HyperModel):
    """Custom hypermodel for Keras Tuner."""

    def __init__(self, num_labels) -> None:
        """Initialise the hypermodel."""
        self.num_labels = num_labels
        self.language_model = None

    def build(self, hp):
        """Build the hypermodel."""
        model_param = {
            "num_labels": self.num_labels,
            "loss_function": LOSS_FUNCTIONS[hp.Fixed("loss_function", "categorical_crossentropy")],
            "optimiser": OPTIMISERS[hp.Fixed("optimiser", "adam")],
            "language_model": hp.Choice(
                "language_model",
                [
                    "distilbert-base-uncased",
                    "distilbert-base-cased",
                    "distilbert-base-multilingual-cased",
                    "GroNLP/bert-base-dutch-cased",
                    "roberta-base",
                ],
            ),
            "use_learning_rate_scheduler": hp.Boolean("use_learning_rate_scheduler"),
            "initial_learning_rate": hp.Fixed(
                "initial_learning_rate",
                0.0004,
                parent_name="use_learning_rate_scheduler",
                parent_values=[True],
            ),
            "decay_steps": hp.Fixed(
                "decay_steps",
                0.0004,
                parent_name="use_learning_rate_scheduler",
                parent_values=[True],
            ),
            "decay_rate": hp.Float(
                "decay_rate",
                max_value=0.95,
                min_value=0.75,
                step=0.05,
                parent_name="use_learning_rate_scheduler",
                parent_values=[True],
            ),
            "learning_rate": hp.Float(
                "learning_rate",
                min_value=0.00005,
                max_value=0.0004,
                step=0.00005,
                parent_name="use_learning_rate_scheduler",
                parent_values=[False],
            ),
        }
        self.language_model = model_param["language_model"]
        return create_model(model_param)

    def fit(self, hp, model, *args, **kwargs):
        """Fit the hypermodel."""
        tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        print(self.language_model)
        print(tokenizer)
        tokens_train = tokenizer(kwargs["x"], padding=True, max_length=100, truncation=True, return_tensors="np").data
        tokens_dev = tokenizer(
            kwargs["validation_data"][0],
            padding=True,
            max_length=100,
            truncation=True,
            return_tensors="np",
        ).data

        return model.fit(
            x=tokens_train,
            y=kwargs["y"],
            validation_data=(tokens_dev, kwargs["validation_data"][1]),
            epochs=kwargs["epochs"],
            batch_size=kwargs["batch_size"],
            callbacks=kwargs["callbacks"],
            verbose=1,
        )

    def test_set_predict(self, X_test, Y_test, ident, encoder):
        """Override to make sure the correct tokenizer is used in the pipeline."""
        tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        tokens_test = tokenizer(X_test, padding=True, max_length=100, truncation=True, return_tensors="np").data
        return test_set_predict(self, tokens_test, Y_test, ident, encoder)


def train_model(model, X_train, Y_train, X_dev, Y_dev, encoder, params):
    """Train the model here. Note the different settings you can experiment with!."""
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = EarlyStopping(monitor="val_loss", patience=params["patience"])
    # Finally fit the model to our data
    model.fit(
        X_train,
        Y_train,
        verbose=verbose,
        epochs=params["epochs"],
        callbacks=[callback],
        batch_size=params["batch_size"],
        validation_data=(X_dev, Y_dev),
    )
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev", encoder)
    return model


def test_set_predict(model, tokens_test, Y_test, ident, encoder):
    """Do predictions and measure accuracy on our own test set (that we split off train)."""
    # Get predictions using the trained model
    Y_pred = model.predict(tokens_test)["logits"]
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print(f"Accuracy on own {ident} set: {round(accuracy_score(Y_test, Y_pred), 3)}")
    # Convert to labels to make classification report more readable
    Y_pred_labels = [encoder.classes_[x] for x in Y_pred]
    Y_test_labels = [encoder.classes_[x] for x in Y_test]
    # Compute and print the precision, recall and F1 score for each class on unseen examples
    print(f"Classification report:\n {classification_report(Y_test_labels, Y_pred_labels)}")
    # Plot a matrix to see which classes the model confuses (where it makes mistakes)
    print(f"Confusion matrix:\n {confusion_matrix(Y_test_labels, Y_pred_labels)}")


def search(args, num_labels, X_train, Y_train_bin, X_dev, Y_dev_bin, encoder):
    """Run the experiment with hyperparameter search."""
    hypermodel = CustomHyperModel(num_labels)
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective="val_accuracy",
        max_trials=args.max_trials,
        directory="lfd_hypermodel_cache_lm",
        project_name="lm",
    )
    print(tuner.search_space_summary())
    tuner.search(
        x=X_train,
        y=Y_train_bin,
        validation_data=(X_dev, Y_dev_bin),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience)],
    )

    print(tuner.results_summary(num_trials=args.max_trials))

    if args.test_file:
        # Rebuild and retrain best model
        best_hp = tuner.get_best_hyperparameters()[0]
        print(f"Best hyperparameter values: {best_hp.values}")
        tokenizer = AutoTokenizer.from_pretrained(best_hp.get_config()["values"]["language_model"])
        tokens_train = tokenizer(X_train, padding=True, max_length=100, truncation=True, return_tensors="np").data
        tokens_dev = tokenizer(X_dev, padding=True, max_length=100, truncation=True, return_tensors="np").data
        model = tuner.hypermodel.build(best_hp)
        model.fit(
            tokens_train,
            Y_train_bin,
            validation_data=(tokens_dev, Y_dev_bin),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience)],
        )

        # Evaluate best model on test data
        run_test(args.test_file, tokenizer, encoder, model)


def single_model(args, num_labels, X_train, Y_train_bin, X_dev, Y_dev_bin, encoder):
    """Run the experiment with a single model."""
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokens_train = tokenizer(X_train, padding=True, max_length=100, truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=100, truncation=True, return_tensors="np").data
    # Create model
    model_param = {
        "num_labels": num_labels,
        "learning_rate": args.learning_rate,
        "loss_function": LOSS_FUNCTIONS[args.loss_function],
        "optimiser": OPTIMISERS[args.optimiser],
        "language_model": args.language_model,
        "use_learning_rate_scheduler": args.learning_rate_scheduler,
        "initial_learning_rate": args.initial_learning_rate,
        "decay_steps": args.decay_steps,
        "end_learning_rate": args.end_learning_rate,
    }

    model = create_model(model_param)

    # Train the model
    train_param = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
    }

    model = train_model(model, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin, encoder, train_param)

    # Do predictions on specified test set
    if args.test_file:
        run_test(args.test_file, tokenizer, encoder, model)


def run_test(test_file, tokenizer, encoder, model):
    """Run the test set on the model."""
    # Read in test set and vectorize
    X_test, Y_test = read_corpus(test_file)
    tokens_test = tokenizer(X_test, padding=True, max_length=100, truncation=True, return_tensors="np").data
    Y_test_bin = encoder.fit_transform(Y_test)
    # Finally do the predictions
    test_set_predict(model, tokens_test, Y_test_bin, "test", encoder)


def main() -> None:
    """Trains and tests neural network given cmd line arguments."""
    args = create_arg_parser()

    # Read in the data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    num_labels = len(set(Y_train))

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    if args.search:
        search(args, num_labels, X_train, Y_train_bin, X_dev, Y_dev_bin, encoder)
    else:
        single_model(args, num_labels, X_train, Y_train_bin, X_dev, Y_dev_bin, encoder)


if __name__ == "__main__":
    main()
