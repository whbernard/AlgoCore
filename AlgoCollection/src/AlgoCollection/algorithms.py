from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def pca(X, n_components=3):
    _pca = PCA(n_components=n_components)
    embeded = _pca.fit_transform(X)
    return embeded


def tsne(X, n_components=3, perplexity=30, learning_rate=200):
    _tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    embeded = _tsne.fit_transform(X)
    return embeded


def umap(X, n_components=3):
    _umap = UMAP(n_components=n_components)
    embedded = _umap.fit_transform(X)
    return embedded
