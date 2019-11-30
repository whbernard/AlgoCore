from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca(X, n_components=2):
    _pca = PCA(n_components=n_components)
    embeded = _pca.fit_transform(X)
    return embeded


def tsne(X, n_components=2, perplexity=30, learning_rate=200):
    _tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    embeded = _tsne.fit_transform(X)
    return embeded
