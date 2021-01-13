import numpy as np
# from metrics import mean_squared_error
# from generate_regression_data import generate_regression_data
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.

        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c

        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf

        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np

            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)


        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.x_train = None
        self.w = None
        self.h = None
        self.confidence = None

    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.


        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        self.h = np.zeros((features.shape[0]))
        X = np.ones((features.size, self.degree + 1))

        for i in range(X.shape[0]):
            for j in range(1, X.shape[1]):
                X[i, j] = features[i]**j

        self.w = np.linalg.inv(X.T @ X) @ X.T @ targets

        for i in range(self.w.size):
            self.h += self.w[i] * np.sort(features)**i

        self.x_train = features  # so that self.visualize can use them

    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        predictions = np.zeros((features.shape[0]))

        for i in range(self.w.size):
            predictions += self.w[i] * features**i

        return predictions

    def visualize(self, features, targets, path='fitted_plot.png', title="Scatter Plot and Fitted Line", color='b'):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        if self.w is None:
            raise ValueError("Must train model first.")

        plt.figure()
        plt.scatter(features, targets, c=color)
        plt.plot(np.sort(self.x_train), self.h, 'g')  # sort features so points plot in order
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig(path)


# degrees = [5]
# amounts = [100]

# for degree in degrees:
#     p = PolynomialRegression(degree)
#     for amount in amounts:
#         x, y = generate_regression_data(degree, amount, amount_of_noise=0.0)
#         p.fit(x, y)
#         plt.clf()
#         p.visualize(x, y)
#         print(x)
#         y_hat = p.predict(x)
#         mse = mean_squared_error(y, y_hat)
#         assert (mse < 1e-1)