import json

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.models import Sequential
from sklearn.tree import plot_tree

from src.student_prediction.utils import data_utils, model_utils


# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(X_train.shape[1], activation='tanh'))
    model.add(Reshape((X_train.shape[1],)))
    return model


# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


# Define the GAN
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def train_gan(X_train, generator, epochs=100, batch_size=32):
    for epoch in range(epochs):
        # Generate random noise as input to the generator
        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate synthetic data
        generated_data = generator.predict(noise)

        # Sample a batch of real data
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]

        # Train the discriminator on real and synthetic data
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))

        # Train the generator via the GAN model
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - D Loss Real: {d_loss_real} - D Loss Fake: {d_loss_fake} - G Loss: {g_loss}")


if __name__ == '__main__':
    # subject = "por"
    subject = "mat"
    # subject = "all"
    dataset_path = f"datasets/student-{subject}.csv"
    # dataset_path = "datasets/student-por.csv"

    df = pd.read_csv(dataset_path)

    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3', 'G2'])
    X_train, y_train, X_test, y_test, preprocessor = data_utils.split_train_test_data(df, y_class="Approved",
                                                                                      test_size=0.3)

    # Create the GAN components
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    # Start GAN training
    train_gan(X_train, generator=generator)

    # Generate new synthetic samples
    noise = np.random.normal(0, 1, (100, 100))
    synthetic_data = generator.predict(noise)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns.to_list())  # Use your actual column names

    # Save to CSV
    synthetic_df.to_csv('datasets/synthetic_data.csv', index=False)

    #     Machine Learning Training Process

    labels = ["Not Approved", "Approved"]
    models = ["RandomForest", "SVM", "LogisticRegression", "DecisionTree", "KNN", "NaiveBayes"]

    metrics_list = []
    for model in models:
        y_pred, trained_model = model_utils.train_model(X_train, y_train, X_test, y_test, model_name=model)
        output_plot_file = f"output/{model}_confusion_matrix.png"
        output_metrics_file = f"output/{model}_metrics.png"
        model_metrics = model_utils.plot_confusion_matrix(test_Y=y_test, y_pred=y_pred, labels=labels, model_name=model,
                                                          output_file=output_plot_file,
                                                          metrics_file=output_metrics_file,
                                                          plot_metrics_figure=False,
                                                          percentage=False)
        metrics_list.append(model_metrics)
        if model == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(trained_model, filled=True, feature_names=df.columns, class_names=labels, rounded=True)
            plt.title("Decision Tree")
            plt.savefig(f"output/{model}_tree.png")
            plt.close()

    with open("../metrics/model_metrics.json", "w") as metrics_json:
        metrics_json.write(json.dumps(metrics_list, indent=2))

    print("finished")
