import numpy as np
from polynomial_regression import PolynomialRegression
from generate_regression_data import generate_regression_data
from metrics import mean_squared_error
from math import log
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


if __name__ == '__main__':

    # QUESTIONS 1 AND 2:

    degree = 4
    N = 100
    x, y = generate_regression_data(degree, N, amount_of_noise=1.0)

    rand_idxs = np.random.choice(N, N, replace=False)
    x_train, y_train = x[rand_idxs[:10]], y[rand_idxs[:10]]
    x_test, y_test = x[rand_idxs[10:]], y[rand_idxs[10:]]

    p_list = []
    mse_train = []
    mse_test = []

    for i in range(9):
        p = PolynomialRegression(i)
        p.fit(x_train, y_train)
        p.visualize(x_train, y_train,
                    path=f'../plots_q1_and_q2/training_plot_degree_{i}',
                    title=f"Training Plot Degree {i}")
        p.visualize(x_test, y_test,
                    path=f'../plots_q1_and_q2/testing_plot_degree_{i}',
                    title=f"Testing Plot Degree {i}",
                    color='r')
        y_hat_train = p.predict(x_train)
        mse_train.append(mean_squared_error(y_train, y_hat_train))
        y_hat_test = p.predict(x_test)
        mse_test.append(mean_squared_error(y_test, y_hat_test))
        p_list.append(p)

    plt.clf()
    plt.figure()
    plt.plot(range(9), [log(mse_train[i]) for i in range(9)], label="training error") 
    plt.plot(range(9), [log(mse_test[i]) for i in range(9)], label="testing error") 
    plt.title("Training and Testing Errors vs. Degree")
    plt.xlabel('degree')
    plt.ylabel("log of error")
    plt.legend()
    plt.grid(True)
    plt.savefig('../plots_q1_and_q2/train_test_errors.png')

    min_train_err_degree = mse_train.index(min(mse_train))
    min_test_err_degree = mse_test.index(min(mse_test))

    plt.clf()
    plt.figure()
    plt.scatter(x_train, y_train)
    plt.plot(np.sort(p_list[min_train_err_degree].x_train),
             p_list[min_train_err_degree].h,
             label=f"min train err curve, degree = {min_train_err_degree}")
    plt.plot(np.sort(p_list[min_test_err_degree].x_train),
             p_list[min_test_err_degree].h,
             label=f"min test err curve, degree = {min_test_err_degree}")
    plt.title("Min. Training and Testing Errors Curves")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('../plots_q1_and_q2/min_train_test_err_curves.png')

    # QUESTION 3

    degree = 4
    N = 100
    x, y = generate_regression_data(degree, N, amount_of_noise=1.0)

    rand_idxs = np.random.choice(N, N, replace=False)
    x_train, y_train = x[rand_idxs[:50]], y[rand_idxs[:50]]
    x_test, y_test = x[rand_idxs[50:]], y[rand_idxs[50:]]

    p_list = []
    mse_train = []
    mse_test = []

    for i in range(9):
        p = PolynomialRegression(i)
        p.fit(x_train, y_train)
        p.visualize(x_train, y_train,
                    path=f'../plots_q3/training_plot_degree_{i}',
                    title=f"Training Plot Degree {i}")
        p.visualize(x_test, y_test,
                    path=f'../plots_q3/testing_plot_degree_{i}',
                    title=f"Testing Plot Degree {i}",
                    color='r')
        y_hat_train = p.predict(x_train)
        mse_train.append(mean_squared_error(y_train, y_hat_train))
        y_hat_test = p.predict(x_test)
        mse_test.append(mean_squared_error(y_test, y_hat_test))
        p_list.append(p)

    plt.clf()
    plt.figure()
    plt.plot(range(9), [log(mse_train[i]) for i in range(9)], label="training error") 
    plt.plot(range(9), [log(mse_test[i]) for i in range(9)], label="testing error") 
    plt.title("Training and Testing Errors vs. Degree")
    plt.xlabel('degree')
    plt.ylabel("log of error")
    plt.legend()
    plt.grid(True)
    plt.savefig('../plots_q3/train_test_errors.png')

    min_train_err_degree = mse_train.index(min(mse_train))
    min_test_err_degree = mse_test.index(min(mse_test))

    plt.clf()
    plt.figure()
    plt.scatter(x_train, y_train)
    plt.plot(np.sort(p_list[min_train_err_degree].x_train),
             p_list[min_train_err_degree].h,
             label=f"min train err curve, degree = {min_train_err_degree}")
    plt.plot(np.sort(p_list[min_test_err_degree].x_train),
             p_list[min_test_err_degree].h,
             label=f"min test err curve, degree = {min_test_err_degree}")
    plt.title("Min. Training and Testing Errors Curves")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('../plots_q3/min_train_test_err_curves.png')
