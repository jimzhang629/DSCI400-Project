from modules.FeatureSelectionTools import csv2df
from modules.FeatureSelectionTools import normalize
from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist

# import data
df = csv2df('~/Documents/DSCI400/DSCI400-Project/src/cached_data/'
            'ALL_DB_COL_data_100_threshold.csv')

# remove duplicates and normalize
# todo: include this data wrangling step as a module
df = df.drop_duplicates()
df = normalize(df)

def elbow(df, lower, upper, fname, mark_idx, step=1):

    ##### cluster data into K=1..20 clusters #####
    KK = range(lower, upper + 1, step)

    KM = [kmeans(df, k, iter=100) for k in KK]
    centroids = [cent for (cent, var) in KM]
    D_k = [cdist(df, cent, 'euclidean') for cent in centroids]
    dist = [np.min(D, axis=1) for D in D_k]

    tot_withinss = [sum(d ** 2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(df) ** 2) / df.shape[0]  # The total sum of squares
    betweenss = totss - tot_withinss  # The between-cluster sum of squares

    # Create a rolling average
    window_width = 2
    cumsum_vec = np.cumsum(np.insert(betweenss, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[
                                          :-window_width]) / window_width

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(KK[window_width - 1:], ma_vec / totss * 100, 'b*-')
    ax.plot(KK[mark_idx], betweenss[mark_idx] / totss * 100, marker='o',
            markersize=12, markeredgewidth=2, markeredgecolor='r',
            markerfacecolor='None')
    ax.set_ylim((60, 100))
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Percentage of Variance Explained (%)')
    plt.title('Elbow for KMeans Clustering')
    plt.savefig(fname, dpi=1000)
    plt.clf()

elbow(df, 1, 475, 'Figures/kmeans elbow.png', 200)