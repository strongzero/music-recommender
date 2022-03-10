"""
contains various implementations for recommending songs
"""

from utils import tracks, feature_matrix, song_label, model_knn
from transform import extract_features, pca_transform
from sklearn.neighbors import NearestNeighbors

def recommend_neighbors(file):  
    """
    Recommend a list of k track ids based on the similarity
    """

    f = extract_features(file)

    # model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    # model_knn.fit(feature_matrix)
    distances, indices = model_knn.kneighbors(pca_transform(f), n_neighbors = 20)

    for i in range(0, len(distances.flatten())):        
        id = feature_matrix.index[indices.flatten()[i]]
        labels = song_label.query('index==@id').to_numpy()
        print('{0}: {1} {2}, with distance of {3}:'.format(i, id, labels, distances.flatten()[i]))      
