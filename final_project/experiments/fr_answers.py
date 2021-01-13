import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from src import (Dog_Classifier_Conv, Synth_Classifier, run_model,
                 load_synth_data)
from data import MyDataset, DogsDataset


def problem_10():

    conv = nn.Conv2d(1, 3, (8, 4), stride=(2, 2), padding=(4, 8),
                     dilation=(1, 1))
    x = torch.Tensor(0.1 * np.ones((16, 1, 32, 32)))
    x = conv(x)

    print(f"Output size: {x.size()}")


def problems_11_and_12():

    kernel_size = [(5, 5), (5, 5)]
    stride = [(1, 1), (1, 1)]

    model = Dog_Classifier_Conv(kernel_size, stride)

    total_params = sum(param.numel() for param in model.parameters())
    print(f'Total num. of weights: {total_params}')

    dataset = DogsDataset('data')

    train_set = MyDataset(dataset.trainX, dataset.trainY)
    valid_set = MyDataset(dataset.validX, dataset.validY)
    test_set = MyDataset(dataset.testX, dataset.testY)

    model, train_valid_loss, train_valid_acc = run_model(
        model, running_mode='train', train_set=train_set, valid_set=valid_set,
        batch_size=10, learning_rate=1e-5, n_epochs=100, shuffle=True)

    print(f"Number of epochs before terminating = {len(train_valid_loss['train'])}")

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
    plt.savefig('experiments/problem_12_loss.png')

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
    plt.savefig('experiments/problem_12_acc.png')


def problems_13_and_14():

    trainX, trainY = load_synth_data('synth_data')

    plt.imshow(trainX[54, :, :])
    plt.colorbar()
    plt.savefig('experiments/problem_13_class_1.png')
    plt.clf()

    plt.imshow(trainX[-1, :, :])
    plt.colorbar()
    plt.savefig('experiments/problem_13_class_2.png')
    plt.clf()

    kernel_size = [(3, 3), (3, 3), (3, 3)]
    stride = [(1, 1), (1, 1), (1, 1)]

    model = Synth_Classifier(kernel_size, stride)

    trainX = np.expand_dims(trainX, axis=3)
    train_set = MyDataset(trainX, trainY)

    print(f"Model params before training: {model.conv1.weight}")

    model, train_loss, train_acc = run_model(
        model, running_mode='train', train_set=train_set, batch_size=50,
        learning_rate=1e-3, n_epochs=50, shuffle=True)

    print(f"Model params after training: {model.conv1.weight}")

    plt.imshow(list(model.conv1.weight)[0].detach().numpy()[0], cmap='gray')
    plt.colorbar()
    plt.savefig('experiments/problem_14_kernel_1.png')
    plt.clf()

    plt.imshow(list(model.conv1.weight)[1].detach().numpy()[0], cmap='gray')
    plt.colorbar()
    plt.savefig('experiments/problem_14_kernel_2.png')
    plt.clf()


if __name__ == '__main__':

    # problem_10()
    # problems_11_and_12()
    problems_13_and_14()
