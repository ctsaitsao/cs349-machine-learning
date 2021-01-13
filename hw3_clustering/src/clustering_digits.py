import numpy as np
from mnist import load_mnist
from metrics import adjusted_mutual_info
from generate_cluster_data import generate_cluster_data
import os
import random
from scipy.stats import multivariate_normal
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

if __name__ == '__main__':

    images, labels = load_mnist(dataset='testing', digits=[1])

    print(images)