import numpy as np
from math import log

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def _fit_helper(self, features, targets, curr_node, attributes):
        if features.size == 0 and targets.size == 0:
            return Node(attribute_name='leaf')

        elif np.count_nonzero(targets == 0) == len(targets):
            return Node(attribute_name='leaf', value=0)

        elif np.count_nonzero(targets == 1) == len(targets):
            return Node(attribute_name='leaf', value=1)

        elif len(attributes) == 0:
            if np.count_nonzero(targets == 1) > np.count_nonzero(targets == 0):
                return Node(attribute_name='leaf', value=1)
            else:
                return Node(attribute_name='leaf', value=0)

        else:
            examples = np.concatenate((features, np.vstack(targets)), 1)
            gain_dict = {}
            for i in range(len(attributes)):
                gain_dict[attributes[i]] = information_gain(features, i, targets)

            curr_node.attribute_name = max(gain_dict, key=gain_dict.get)
            curr_node.attribute_index = self.attribute_names.index(curr_node.attribute_name)
            attribute_index_mutable = attributes.index(curr_node.attribute_name)
            attributes_copy = attributes.copy()
            attributes_copy.remove(curr_node.attribute_name)

            left_child_examples = examples[examples[:, attribute_index_mutable] == 0] 
            left_child_features = left_child_examples[:,:-1]
            left_child_features = np.delete(left_child_features, attribute_index_mutable, 1)
            left_child_targets = left_child_examples[:, -1]

            right_child_examples = examples[examples[:, attribute_index_mutable] == 1]
            right_child_features = right_child_examples[:,:-1]
            right_child_features = np.delete(right_child_features, attribute_index_mutable, 1)
            right_child_targets = right_child_examples[:, -1]

            left_child = self._fit_helper(left_child_features, left_child_targets, Node(), attributes_copy)
            right_child = self._fit_helper(right_child_features, right_child_targets, Node(), attributes_copy)

            curr_node.branches.append(left_child)
            curr_node.branches.append(right_child)

            return curr_node  

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        self.tree = self._fit_helper(features, targets, Node(), self.attribute_names)

    def _predict_helper(self, point, curr_node):
        if curr_node.branches == []:
            return curr_node.value

        else:
            if (point[curr_node.attribute_index] == 0):
                return self._predict_helper(point, curr_node.branches[0])
            else:
                return self._predict_helper(point, curr_node.branches[1])

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        self._check_input(features)
        predictions = np.zeros((features.shape[0]))

        i = 0
        for point in features:
            prediction = self._predict_helper(point, self.tree)
            predictions[i] = prediction
            i += 1

        return predictions

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    attribute_column = features[:, attribute_index]
    attribute_labels = np.unique(attribute_column)
    target_labels = np.unique(targets)

    weights_dict = {}
    priors_dict = {}

    for attribute_label in attribute_labels:
        weights_dict[attribute_label] = np.count_nonzero(attribute_column == attribute_label) / attribute_column.size
        priors_dict[attribute_label] = {}

        attribute_label_indices = np.argwhere(attribute_column == attribute_label)
        attribute_targets = np.zeros((len(attribute_label_indices)))
        i = 0
        for index in attribute_label_indices:
            attribute_targets[i] = targets[index[0]]   
            i += 1

        for target_label in target_labels:
            priors_dict[attribute_label][target_label] = np.count_nonzero(attribute_targets == target_label) \
                / np.count_nonzero(attribute_column == attribute_label)

    entropy_attribute = 0

    for attribute_label in priors_dict.keys():
        label_entropy = 0
        for prior in priors_dict[attribute_label].values():
            label_entropy -= 0 if prior == 0 else prior * log(prior, 2)
        entropy_attribute += weights_dict[attribute_label] * label_entropy

    entropy_set = 0

    for target_label in target_labels:
        prior = np.count_nonzero(targets == target_label) / targets.size
        entropy_set -= prior * log(prior, 2)

    information_gain = entropy_set - entropy_attribute
    return information_gain

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    # features, targets, attribute_names = load_data('../data/ivy-league.csv')
    decision_tree = DecisionTree(attribute_names=attribute_names)
    # decision_tree.fit(features, targets)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()