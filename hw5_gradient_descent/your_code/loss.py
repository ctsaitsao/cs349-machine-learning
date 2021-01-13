import numpy as np


class Loss:
    """
    An abstract base class for a loss function that computes both the prescribed
    loss function (the forward pass) as well as its gradient (the backward
    pass).

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        regularization - (`Regularization` or None) The type of regularization to
            perform. Either a derived class of `Regularization` or None. If None,
            no regularization is performed.
    """

    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        pass

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_s(x, y; w) = (1/2) (y - w^T x)^2

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        ## METHOD TOO SLOW: ##
        # N = X.shape[0]
        # loss_arr = np.zeros((N))

        # for i in range(N):  
        #     loss_arr[i] = 1/2 * (y[i] - np.atleast_2d(w) @ np.vstack(X[i, :]))**2

        # loss = np.sum(loss_arr) / N

        loss = 1/(2*X.shape[0]) * np.sum((y - np.array([np.dot(w.T, feature) for feature in X]))**2)

        if self.regularization:
            loss += self.regularization.forward(w)

        return loss

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        ## METHOD TOO SLOW: ##
        # N = X.shape[0]
        # gradient_mat = np.zeros((X.shape))

        # for i in range(N):
        #     gradient_mat[i, :] = (y[i] - np.atleast_2d(w) @ np.vstack(X[i, :])) * X[i, :]

        # gradient = np.sum(gradient_mat, 0) / -N

        ## ALSO SLOW: ##
        # gradient = -1/X.shape[0] * np.sum(np.array([(y - np.array([np.dot(w.T, feature) for feature in X]))[i] * X[i, :] for i in range(X.shape[0])]), 0)

        mult = np.array([np.dot(w.T, feature) for feature in X])
        sub = y - mult
        sub_arr = np.array([sub[i] * X[i, :] for i in range(X.shape[0])])
        thing_to_sum = np.sum(sub_arr, 0)
        gradient = -1/X.shape[0] * thing_to_sum

        if self.regularization:
            gradient += self.regularization.backward(w)

        return gradient


class HingeLoss(Loss):
    """
    The hinge loss function.

    https://en.wikipedia.org/wiki/Hinge_loss
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The hinge loss for a single example
        is given as follows:

        L_h(x, y; w) = max(0, 1 - y w^T x)

        The hinge loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        ## METHOD TOO SLOW: ##
        # N = X.shape[0]
        # loss_arr = np.zeros((N))

        # for i in range(N):
        #     loss_arr[i] = max(0, 1 - y[i] * np.atleast_2d(w) @ np.vstack(X[i, :]))

        # loss = np.sum(loss_arr) / N

        thing_to_sum = 1 - y * np.array([np.dot(w.T, feature) for feature in X])
        loss = 1/X.shape[0] * np.sum(np.where(thing_to_sum < 0, 0, thing_to_sum))

        if self.regularization:
            loss += self.regularization.forward(w)

        return loss

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        ## METHOD TOO SLOW: ##
        # N = X.shape[0]
        # gradient_mat = np.zeros((X.shape))

        # for i in range(N):
        #     gradient_mat[i, :] = (1 - y[i] * np.atleast_2d(w) @ np.vstack(X[i, :]) > 0) \
        #                             * -y[i] * X[i, :]

        # gradient = np.sum(gradient_mat, 0) / N

        # if self.regularization:
        #     gradient += self.regularization.backward(w)

        gradient = 1/X.shape[0] * np.sum(np.array([(-y[i] * X[i, :]) if ((1 - y[i] * np.dot(w.T, X[i, :])) > 0) else np.zeros(X.shape[1]) for i in range(X.shape[0])]), 0)

        if self.regularization:
            gradient += self.regularization.backward(w)

        return gradient


class ZeroOneLoss(Loss):
    """
    The 0-1 loss function.

    The loss is 0 iff w^T x == y, else the loss is 1.

    *** YOU DO NOT NEED TO IMPLEMENT THIS ***
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_0-1(x, y; w) = {0 iff w^T x == y, else 1}

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        if self.regularization:
            loss += self.regularization.forward(w)
        return loss

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        # This function purposefully left blank
        raise ValueError('No need to use this function for the homework :p')
