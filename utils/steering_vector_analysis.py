import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def plot_PCA(steering_vectors: dict[str: torch.Tensor]) -> None:
    '''
    plots steering vector on a 2d PCA plot

    Args:
        steering_vectors: a dictionary with keys as languages, and values as torch Tensor steering vectors
    '''

    languages, vectors = zip(*steering_vectors.items())
    vectors = torch.stack(vectors).cpu().numpy()

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(vectors)

    x, y = transformed.T 
    scatter = plt.scatter(x, y, c=range(len(x)), cmap='viridis')
    plt.legend(handles=scatter.legend_elements()[0], labels=languages)




