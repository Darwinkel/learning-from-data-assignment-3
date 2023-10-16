#!/usr/bin/env python3.11
"""Run the second part of assignment 3 with LSTM-based models."""

import argparse
import json
import random as python_random

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import LSTM, Bidirectional, Dense, Embedding, TextVectorization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

optimisers = {
    "sgd": SGD,
    "adam": Adam,
}


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
    parser.add_argument(
        "-e",
        "--embeddings",
        default="data/glove_reviews.json",
        type=str,
        help="Embedding file we are using (default data/glove_reviews.json)",
    )
    parser.add_argument("-lr", "--learning_rate", default=1e-2, type=float, help="Learning rate")
    parser.add_argument("-ep", "--epochs", default=60, type=int, help="Number of training epochs")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("-p", "--patience", default=3, type=int, help="Amount of epochs before early stopping")
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
    parser.add_argument("-do", "--dropout", default=0.0, type=float, help="Fraction of (input) dropout")
    parser.add_argument(
        "-rdo",
        "--recurrent_dropout",
        default=0.0,
        type=float,
        help="Fraction of recurrent-state dropout",
    )
    parser.add_argument("-bd", "--bidirectional", action="store_true", help="Use a bidirectional LSTM")
    parser.add_argument("-dl", "--dense", action="store_true", help="Add a dense layer between embedding and LSTM")
    parser.add_argument("-te", "--trainable", action="store_true", help="Use trainable embeddings")
    parser.add_argument("-l", "--num_layers", default=1, type=int, help="Number of LSTM layers")
    parser.add_argument(
        "-a",
        "--activation",
        default="tanh",
        type=str,
        help="Activation function to use for LSTM (default tanh)",
    )
    parser.add_argument(
        "-ra",
        "--recurrent_activation",
        default="sigmoid",
        type=str,
        help="Activation function to use for LSTM recurrent states (default sigmoid)",
    )
    parser.add_argument(
        "-s",
        "--search",
        action="store_true",
        help="Perform full hyperparameter search (warning: takes a while!)",
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


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array."""
    embeddings = json.load(open(embeddings_file))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings."""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix, params):
    """Create the Keras model to use."""
    # Define settings, you might want to create cmd line args for them
    optim = params["optimiser"](learning_rate=params["learning_rate"])
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))
    # Now build the model
    model = Sequential()
    model.add(
        Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=Constant(emb_matrix),
            trainable=params["trainable"],
        ),
    )
    if params["dense"]:
        model.add(Dense(300, activation=params["activation"]))
    for i in range(params["num_layers"]):
        # Return sequences for all LSTM layers except last
        ret_seq = i != (params["num_layers"] - 1)
        if params["bidirectional"]:
            model.add(Bidirectional(LSTM(300, return_sequences=ret_seq)))
        else:
            model.add(LSTM(300, return_sequences=ret_seq))
    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=params["loss_function"], optimizer=optim, metrics=["accuracy"])
    print(model.summary())
    return model


class CustomHyperModel(kt.HyperModel):
    """Custom hypermodel for Keras Tuner."""

    def __init__(self, Y_train, emb_matrix, epochs, X_dev, Y_dev, patience) -> None:
        """Initialise the hypermodel."""
        self.Y_train = Y_train
        self.emb_matrix = emb_matrix
        self.epochs = epochs
        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.patience = patience

    def build(self, hp):
        """Build the hypermodel."""
        params = {
            "learning_rate": hp.Float("learning_rate", min_value=1e-4, max_value=1e-1, step=10, sampling="log"),
            "loss_function": hp.Choice("loss_function", ["categorical_crossentropy"]),
            "optimiser": optimisers[hp.Choice("optimiser", ["sgd", "adam"])],
            "dropout": hp.Float("dropout", min_value=0.0, max_value=0.2, step=0.05),
            "rec_dropout": hp.Float("recurrent_dropout", min_value=0.0, max_value=0.2, step=0.05),
            "bidirectional": hp.Boolean("bidirectional"),
            "trainable": hp.Boolean("trainable"),
            "dense": hp.Boolean("dense"),
            "num_layers": hp.Int("num_layers", min_value=1, max_value=3, step=1),
            "activation": hp.Choice("activation", ["tanh", "sigmoid"]),
            "rec_activation": hp.Choice("recurrent_activation", ["tanh", "sigmoid"]),
        }
        return create_model(self.Y_train, self.emb_matrix, params)

    def fit(self, hp, model, *args, **kwargs):
        """Fit the hypermodel."""
        return model.fit(
            *args,
            verbose=1,
            **kwargs,
        )


def train_model(model, X_train, Y_train, X_dev, Y_dev, batch_size, epochs, encoder, patience):
    """Train the model here. Note the different settings you can experiment with!."""
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = EarlyStopping(monitor="val_loss", patience=patience)
    # Finally fit the model to our data
    model.fit(
        X_train,
        Y_train,
        verbose=verbose,
        epochs=epochs,
        callbacks=[callback],
        batch_size=batch_size,
        validation_data=(X_dev, Y_dev),
    )
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev", encoder)
    return model


def test_set_predict(model, X_test, Y_test, ident, encoder):
    """Do predictions and measure accuracy on our own test set (that we split off train)."""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
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


def search(
    args,
    Y_train,
    emb_matrix,
    X_train_vect,
    Y_train_bin,
    X_dev_vect,
    Y_dev_bin,
    encoder,
    vectorizer,
    X_dev,
    Y_dev,
):
    """Run the experiment with hyperparameter search."""
    hypermodel = CustomHyperModel(Y_train, emb_matrix, args.epochs, X_dev, Y_dev, args.patience)
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective="val_accuracy",
        max_trials=args.max_trials,
        directory="lfd_hypermodel_cache",
        project_name="lstm",
    )
    print(tuner.search_space_summary())
    tuner.search(
        X_train_vect,
        Y_train_bin,
        validation_data=(X_dev_vect, Y_dev_bin),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience)],
    )

    print(tuner.results_summary(num_trials=args.max_trials))

    if args.test_file:
        # Rebuild and retrain best model
        best_hp = tuner.get_best_hyperparameters()[0]
        print(f"Best hyperparameter values: {best_hp.values}")
        model = tuner.hypermodel.build(best_hp)
        model = train_model(
            model,
            X_train_vect,
            Y_train_bin,
            X_dev_vect,
            Y_dev_bin,
            args.batch_size,
            args.epochs,
            encoder,
            args.patience,
        )

        run_test(args.test_file, vectorizer, encoder, model)


def run_test(test_file, vectorizer, encoder, model):
    """Run the test set on the model."""
    # Read in test set and vectorize
    X_test, Y_test = read_corpus(test_file)
    Y_test_bin = encoder.fit_transform(Y_test)
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
    # Finally do the predictions
    test_set_predict(model, X_test_vect, Y_test_bin, "test", encoder)


def single_model(args, Y_train, emb_matrix, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, encoder, vectorizer):
    """Run the experiment with a single model."""
    # Create model
    model_param = {
        "learning_rate": args.learning_rate,
        "loss_function": args.loss_function,
        "optimiser": optimisers[args.optimiser],
        "dropout": args.dropout,
        "rec_dropout": args.recurrent_dropout,
        "bidirectional": args.bidirectional,
        "trainable": args.trainable,
        "dense": args.dense,
        "num_layers": args.num_layers,
        "activation": args.activation,
        "rec_activation": args.recurrent_activation,
    }

    model = create_model(Y_train, emb_matrix, model_param)

    # Train the model
    model = train_model(
        model,
        X_train_vect,
        Y_train_bin,
        X_dev_vect,
        Y_dev_bin,
        args.batch_size,
        args.epochs,
        encoder,
        args.patience,
    )

    # Do predictions on specified test set
    if args.test_file:
        run_test(args.test_file, vectorizer, encoder, model)


def main() -> None:
    """Trains and tests neural network given cmd line arguments."""
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer

    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    if args.search:
        search(
            args,
            Y_train,
            emb_matrix,
            X_train_vect,
            Y_train_bin,
            X_dev_vect,
            Y_dev_bin,
            encoder,
            vectorizer,
            X_dev,
            Y_dev,
        )
    else:
        single_model(args, Y_train, emb_matrix, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, encoder, vectorizer)


if __name__ == "__main__":
    main()
