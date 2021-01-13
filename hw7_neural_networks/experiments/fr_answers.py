import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from src import Digit_Classifier, Dog_Classifier_FC, run_model
from data import load_mnist_data, MyDataset, DogsDataset


def problems_1_and_3():

    n_examples_list = [500, 1000, 1500, 2000]
    time_list = []
    test_acc_list = []

    for n_examples in n_examples_list:

        model = Digit_Classifier()

        train_features, _, train_targets, _ = load_mnist_data(
            10, fraction=1.0, examples_per_class=int(n_examples / 10))

        _, test_features, _, test_targets = load_mnist_data(
            10, fraction=0.0, examples_per_class=int(1000 / 10))

        train_set = MyDataset(train_features, train_targets)
        test_set = MyDataset(test_features, test_targets)

        start_time = time.time()

        model, _, _ = run_model(model, running_mode='train',
                                train_set=train_set, batch_size=10,
                                n_epochs=100, shuffle=True)

        time_list.append(time.time() - start_time)

        _, acc = run_model(model, running_mode='test', test_set=test_set,
                           batch_size=10, n_epochs=100, shuffle=True)

        test_acc_list.append(acc)

    plt.figure()
    plt.plot(n_examples_list, time_list, '--bo')
    plt.title("Training Time vs. Num. Training Examples")
    plt.xlabel('Num. Training Examples')
    plt.ylabel("Training Time for 100 epochs (s)")
    plt.grid(True)
    plt.savefig('experiments/problem_1.png')

    plt.figure()
    plt.plot(n_examples_list, test_acc_list, '--bo')
    plt.title("Testing Accuracy vs. Num. Training Examples")
    plt.xlabel('Num. Training Examples')
    plt.ylabel("Testing Accuracy (%)")
    plt.grid(True)
    plt.savefig('experiments/problem_3.png')


def problem_8():

    model = Dog_Classifier_FC()
    dataset = DogsDataset('data')

    train_set = MyDataset(dataset.trainX, dataset.trainY)
    valid_set = MyDataset(dataset.validX, dataset.validY)
    test_set = MyDataset(dataset.testX, dataset.testY)

    # For question 5:
    print(f"train_set size = {dataset.trainY.size}")
    print(f"valid_set size = {dataset.validY.size}")
    print(f"test_set size = {dataset.testY.size}")

    model, train_valid_loss, train_valid_acc = run_model(
        model, running_mode='train', train_set=train_set, valid_set=valid_set,
        batch_size=10, learning_rate=1e-5, n_epochs=100, shuffle=True)

    print(f"Number of epochs before terminating = {len(train_valid_loss['train'])}")

    total_params = sum(param.numel() for param in model.parameters())
    print(f'Total num. of weights: {total_params}')

    _, test_acc = run_model(model, running_mode='test',
                            test_set=test_set, batch_size=10,
                            learning_rate=1e-5, n_epochs=100,
                            shuffle=True)

    print(f"Accuracy on testing set = {test_acc}")

    plt.figure()
    plt.plot(range(len(train_valid_loss['train'])), train_valid_loss['train'],
             label='training loss')
    plt.plot(range(len(train_valid_loss['valid'])), train_valid_loss['valid'],
             label='validation loss')
    plt.legend()
    plt.title("Training and Validation Loss vs. Num. Epochs")
    plt.xlabel('Num. Epochs')
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('experiments/problem_8_loss.png')

    plt.figure()
    plt.plot(range(len(train_valid_acc['train'])), train_valid_acc['train'],
             label='training accuracy')
    plt.plot(range(len(train_valid_acc['valid'])), train_valid_acc['valid'],
             label='validation accuracy')
    plt.legend()
    plt.title("Training and Validation Accuracy vs. Num. Epochs")
    plt.xlabel('Num. Epochs')
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig('experiments/problem_8_acc.png')


if __name__ == '__main__':

    # problems_1_and_3()
    problem_8()