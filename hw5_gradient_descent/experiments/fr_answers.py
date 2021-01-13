from your_code import L1Regularization, L2Regularization
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import GradientDescent
from your_code import MultiClassGradientDescent
from your_code import accuracy, confusion_matrix
from your_code import load_data
import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap


if __name__ == "__main__":

    # # Problem 1a:

    # train_features, _, train_targets, _ = load_data('mnist-binary', fraction=1.0)

    # gd = GradientDescent(loss='hinge', learning_rate=1e-4)
    # gd.fit(train_features, train_targets)

    # plt.figure()
    # plt.plot(range(len(gd.loss_list)), gd.loss_list)
    # plt.xlabel('iteration')
    # plt.ylabel('loss')
    # plt.title('Loss vs. Iteration')
    # plt.savefig('experiments/1a_loss.png')

    # plt.figure()
    # plt.plot(range(len(gd.accuracy_list)), gd.accuracy_list)
    # plt.xlabel('iteration')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy vs. Iteration')
    # plt.savefig('experiments/1a_acc.png')

    # # Problem 1b:

    # train_features, _, train_targets, _ = load_data('mnist-binary', fraction=1.0)

    # gd = GradientDescent(loss='hinge', learning_rate=1e-4)

    # loss_at_epoch = []
    # acc_at_epoch = []

    # gd.fit(train_features, train_targets, batch_size=1, max_iter=1000*train_features.shape[0])
    # # loss_at_epoch.append(gd.loss_list[0])
    # # loss_at_epoch.append(gd.accuracy_list[0])

    # plt.figure()
    # plt.plot(range(len(gd.loss_list)), gd.loss_list)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('Loss vs. Epoch')
    # plt.savefig('experiments/1b_loss.png')

    # plt.figure()
    # plt.plot(range(len(gd.accuracy_list)), gd.accuracy_list)
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy vs. Epoch')
    # plt.savefig('experiments/1b_acc.png')

    # # Problem 2a:

    # train_features, _, train_targets, _ = load_data('synthetic', fraction=1.0)

    # bias_list = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
    # loss_list = []

    # w0 = np.ones(train_features.shape[1])
    # train_features = np.append(train_features, np.ones((len(train_features), 1)), axis=1)

    # for bias in bias_list:
    #     zol = ZeroOneLoss()
    #     w = np.append(w0, bias)
    #     loss = zol.forward(train_features, w, train_targets)
    #     loss_list.append(loss)

    # plt.figure()
    # plt.plot(bias_list, loss_list)
    # plt.title('Loss vs. Bias (Loss Landscape)')
    # plt.xlabel('bias')
    # plt.ylabel('loss')
    # plt.savefig('experiments/2a.png')

    # # Problem 2b:

    # train_features, _, train_targets, _ = load_data('synthetic', fraction=1.0)

    # train_features = train_features[[0, 1, 3, 4]]
    # train_targets = train_targets[[0, 1, 3, 4]]

    # bias_list = [0.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
    # loss_list = []

    # w0 = np.ones(train_features.shape[1])
    # train_features = np.append(train_features, np.ones((len(train_features), 1)), axis=1)

    # for bias in bias_list:
    #     zol = ZeroOneLoss()
    #     w = np.append(w0, bias)
    #     loss = zol.forward(train_features, w, train_targets)
    #     loss_list.append(loss)

    # plt.figure()
    # plt.plot(bias_list, loss_list)
    # plt.title('Loss vs. Bias (Loss Landscape)')
    # plt.xlabel('bias')
    # plt.ylabel('loss')
    # plt.savefig('experiments/2b.png')

    # # Problem 3a:

    # train_features, test_features, train_targets, test_targets = \
    #     load_data('mnist-multiclass', fraction=0.75)

    # mcgd = MultiClassGradientDescent(loss='squared', regularization='l1')
    # mcgd.fit(train_features, train_targets)
    # predictions = mcgd.predict(test_features)

    # confusion_matrix = confusion_matrix(test_targets, predictions)
    # print(f'Confusion Matrix:\n {confusion_matrix}')
    # np.savetxt('experiments/confusion_matrix.csv', confusion_matrix, delimiter=",")

    # # Problem 4a:

    # train_features, test_features, train_targets, test_targets = \
    #     load_data('mnist-binary', fraction=1.0)

    # lambda_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    # l1_list = []
    # l2_list = []

    # for lambda_ in lambda_list:

    #     gd_l1 = GradientDescent('squared', regularization='l1',
    #                             learning_rate=1e-5, reg_param=lambda_)
    #     gd_l2 = GradientDescent('squared', regularization='l2',
    #                             learning_rate=1e-5, reg_param=lambda_)

    #     gd_l1.fit(train_features, train_targets, max_iter=2000)
    #     gd_l2.fit(train_features, train_targets, max_iter=2000)

    #     l1_list.append(np.sum(np.where(abs(gd_l1.model) > 1e-3, 1, 0)))
    #     l2_list.append(np.sum(np.where(abs(gd_l2.model) > 1e-3, 1, 0)))

    # plt.figure()
    # plt.plot(range(len(lambda_list)), l1_list, label='l1')
    # plt.plot(range(len(lambda_list)), l2_list, label='l2')
    # plt.legend()
    # plt.title('# Nonzero Weights vs. Lambda')
    # plt.xlabel('lambda')
    # plt.xticks([0,1,2,3,4,5], ['1e-3', '1e-2', '1e-1', '1', '10', '100'])
    # plt.ylabel('# nonzero w')
    # plt.savefig('experiments/4a.png')

    # # Problem 4c:

    # train_features, test_features, train_targets, test_targets = \
    #     load_data('mnist-binary', fraction=1.0)

    # gd_l1 = GradientDescent(loss='squared', regularization='l1',
    #                         learning_rate=1e-5, reg_param=1)
    # gd_l1.fit(train_features, train_targets, max_iter=2000)

    # hmap = np.where(abs(gd_l1.model) > 1e-3, 0, 1)
    # hmap = np.reshape(hmap[:-1], (train_features.shape))
    # # hmap = heatmap(hmap)
    # plt.imsave('experiments/4c.png', hmap)
    # # hmap.savefig('experiments/4c.png')