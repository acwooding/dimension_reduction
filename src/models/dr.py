#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.manifold import TSNE
from umap import UMAP


DR_ALGORITHMS = {
    "autoencoder": None,
    "HLLE": LocallyLinearEmbedding(method='hessian'),
    "Isomap": Isomap(),
    "KernelPCA": KernelPCA(),
    "LaplacianEigenmaps": SpectralEmbedding(),
    "LLE": LocallyLinearEmbedding(),
    "LTSA": LocallyLinearEmbedding(method='ltsa'),
    "MDS": MDS(),
    "PCA": PCA(),
    "TSNE": TSNE(),
    "UMAP": UMAP(),
}

def available_algorithms():
    """Valid Algorithms for dimension reduction applications

    This function simply returns the list of known dimension reduction
    algorithms.

    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid quality metrics, and the function they map to, are:

    ============     ====================================
    Algorithm        Function
    ============     ====================================
    autoencoder
    isomap
    MDS
    PCA
    t-SNE
    UMAP
    ============     ====================================
    """
    return DR_ALGORITHMS


