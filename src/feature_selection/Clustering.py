from src.modules.DataWrangling import wrangle
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist

# import data
df = wrangle('../cached_data/ALL_DB_COL_data_100_threshold.csv')

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
    KK = range(lower, upper + 1, step)

    # Calculate the distances, entropy, and variance
    KM = [kmeans(df, k, iter=100) for k in KK]
    centroids = [cent for (cent, var) in KM]
    D_k = [cdist(df, cent, 'euclidean') for cent in centroids]
    dist = [np.min(D, axis=1) for D in D_k]
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
    ax.plot(KK[window_width - 1:], ma_vec / totss * 100, 'b*-')
    ax.plot(KK[k_idx], betweenss[k_idx] / totss * 100, marker='o',
            markersize=12, markeredgewidth=2, markeredgecolor='r',
            markerfacecolor='None')
    ax.set_ylim((60, 100))
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Percentage of Variance Explained (%)')
    plt.title('Elbow for KMeans Clustering')
    plt.savefig(fname, dpi=1000)
    plt.clf()

    return KM[int(step/k_idx)]


# knn = elbow(df, 25, 475, 'Figures/kmeans elbow.png', 5, 25)

# knn2 = kmeans2(df, 150, iter=10000, minit='++')[1]
kmeans = KMeans(n_clusters=150, random_state=0, n_init=5000,
                max_iter=10000).fit(df)  # random_state is seed

