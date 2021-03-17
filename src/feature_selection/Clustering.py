import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist


def elbow(df, lower, upper, fname, k_idx, step=1):
    """
    Plots the elbow curve of K-means for multiple values of K.
    @param df: The DataFrame object containing all data.
    @param lower: The lower bound for K.
    @param upper: the upper bound for K.
    @param fname: The filename for the figure to export.
    @param k_idx: The optimal index of K to plot and to return.
    @param step: The step size for K
    @return: The centroid clusters after running K-means with k_idx = K.
    """
    if upper > len(df):
        raise Exception('Upper neighbors bound higher than feature count')

    # Cluster the data with multiple K values
    kk = range(lower, upper + 1, step)

    # Calculate the distances, entropy, and variance
    km = [kmeans(df, k, iter=100) for k in kk]
    centroids = [cent for (cent, var) in km]
    d_k = [cdist(df, cent, 'euclidean') for cent in centroids]
    dist = [np.min(D, axis=1) for D in d_k]
    tot_withinss = [sum(d ** 2) for d in dist]
    totss = sum(pdist(df) ** 2) / df.shape[0]  # The total sum of squares
    betweenss = totss - tot_withinss  # The between-cluster sum of squares

    # Create a rolling average to reduce noise
    window_width = 2
    cumsum_vec = np.cumsum(np.insert(betweenss, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[
                                          :-window_width]) / window_width

    # Plot the elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kk[window_width - 1:], ma_vec / totss * 100, 'b*-')
    ax.plot(kk[k_idx], betweenss[k_idx] / totss * 100, marker='o',
            markersize=12, markeredgewidth=2, markeredgecolor='r',
            markerfacecolor='None')
    ax.set_ylim((60, 100))
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Percentage of Variance Explained (%)')
    plt.title('Elbow for KMeans Clustering')
    plt.savefig(fname, dpi=1000)
    plt.clf()

    return km[int(step / k_idx)]


def k_means(df, n, n_init, max_iter):
    """
    Performs k-means clustering on the input data.
    @param df: The input DataFrame. The rows are the features.
    @param n: The number of resulting clusters.
    @param n_init: The number of time the k-means algorithm will be run with
    different centroid seeds.
    @param max_iter: The maximum number of iterations of the k-means algorithm
    for a single run
    @return: The fitted estimator, the coordinates of the centroids, a
    mapping of each feature to a cluster, and the inertia of the fit.
    """

    # use scikitklearn kmeans since scipy kmeans does not return clusters
    # and labels
    km = KMeans(n_clusters=n, random_state=0, n_init=n_init,
                max_iter=max_iter).fit(df)
    centroids = km.cluster_centers_
    labels = km.labels_
    inertia = km.inertia_
    return km, pd.DataFrame(centroids), labels, inertia


def plot_random_clusters(df, centroids, labels, save):
    """
    Plots 4 randomly selected centroids and all features that belong to that
    cluster.
    @param df: The DataFrame of original data prior to running k-means.
    @param centroids: The coordinates of the centroids.
    @param labels: A mapping of each feature to a cluster.
    @param n: The number of clusters and associated features to plot.

    """
    # Generate 4 random integers
    ran_centroids = random.sample(range(1, len(centroids)), 4)

    # Plot 4 subplots of the clusters
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax_order = [ax1, ax2, ax3, ax4]  # define order for iteration
    fig.suptitle('4 Random Clusters and Centroids')

    # Plot centroid and associated features for each random integer
    for i in range(0, len(ran_centroids)):

        # find the features that were assigned to the centroid
        idx = []
        for k, l in enumerate(labels):
            if l == ran_centroids[i]:
                idx.append(k)

        for feature in idx:
            ax_order[i].plot(df.iloc[feature], 'tab:blue')

        # centroid plotted in black
        ax_order[i].plot(centroids.iloc[ran_centroids[i]], 'tab:red')

    # remove tick marks and only label outer plots
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    for ax in fig.get_axes():
        ax.set(xlabel='Year', ylabel='Indicator Value')
        ax.label_outer()

    # save image in Figures subdirectory if save==True
    if save:
        plt.savefig('src/feature_selection/Figures/clustering_subplot.png',
                    dpi=1000)
