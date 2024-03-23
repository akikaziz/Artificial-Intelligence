import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):

    data = []

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))

    return data

def calc_features(row):

    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])

    return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)


def hac(features):

    n = len(features)
    clusters = [{'ind': i, 'clusters': [i]} for i in range(n)]
    dist = np.zeros((n, n))

    for i in  range(n):
        for j in range(i + 1, n):

            dist[i,j] = np.linalg.norm(features[i] - features[j])
            dist[j,i] = dist[i,j]

    Z = np.zeros((n - 1, 4))

    for k in range(n-1):

        x = -1
        y = -1
        distMin = np.inf

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                distMax = max(dist[c1, c2] for c1 in clusters[i]['clusters'] for c2 in clusters[j]['clusters'])

                if distMax < distMin:
                    distMin = distMax
                    x = i
                    y = j

        Z[k, 0] = clusters[x]['ind']
        Z[k, 1] = clusters[y]['ind']
        Z[k, 2] = distMin
        Z[k, 3] = len(clusters[x]['clusters']) + len(clusters[y]['clusters'])

        updatedClusterIndex = n + k
        dist = np.pad(dist, ((0, 1), (0, 1)), mode = 'constant')

        for p in range(len(clusters)):

            if clusters[p]['ind'] == clusters[x]['ind'] or clusters[p]['ind'] == clusters[y]['ind']:
                dist[p, updatedClusterIndex] = distMin
                dist[updatedClusterIndex, p] = distMin

        newCluster = {'ind': n + k, 'clusters': clusters[x]['clusters'] + clusters[y]['clusters']}

        clusters.pop(x)
        clusters.pop(y - 1)
        clusters.append(newCluster)

    return Z

def fig_hac(Z, names):

    fig = plt.figure()
    dendrogram(Z, labels = names, leaf_rotation = 90)
    fig.tight_layout()

    return fig

def normalize_features(features):

    normalized = []

    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    for row in features:
        normalized_row = (row - means)/stds
        normalized.append(normalized_row)

    return normalized

def main():

    data = load_data('countries.csv')
    country_names = [row['Country'] for row in data] 
    features = [calc_features(row) for row in data] 
    features_normalized = normalize_features(features) 
    n = 10
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n]) 
    fig = fig_hac(Z_raw, country_names[:n]) 
    plt.show()

if __name__=="__main__":
    main()