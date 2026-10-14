# pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
sns.set()
from sklearn.decomposition import PCA

def get_data():
    """生成聚类数据"""
    from sklearn.datasets import make_blobs
    x_value, y_value = make_blobs(n_samples=1000, n_features=40, centers=3, )
    return x_value, y_value


def plot_xy(x_values, label, title):
    """绘图"""
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label
    sns.scatterplot(x="x", y="y", hue="label", data=df)
    plt.title(title)
    plt.show()
import seaborn as sns
from matplotlib.colors import ListedColormap
def draw_simple_tsne_(x_value,y_value):
    """
    feature must be given in row vector
    :param x_value:feature
    :param y_value:label
    :return:
    """

    tsne = TSNE(n_components=2, n_iter=10000,perplexity=10, random_state=42)
    x_tsne = tsne.fit_transform(x_value)
    # plot_xy(x_tsne, y_value, "t-sne",cmap=custom_colors)

    # y_value=y_value//2

    palette = sns.color_palette("pastel", n_colors=10) #palette=palette,
    sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=y_value,  s=50)
    plt.title("t-SNE with Seaborn Palette")
    plt.show()


def draw_simple_tsne_NMC(x_value,y_value):
    """
    feature must be given in row vector
    :param x_value:feature
    :param y_value:label
    :return:
    """


    tsne = TSNE(n_components=2, n_iter=3000,perplexity=20,)
    x_tsne = tsne.fit_transform(x_value)
    plot_xy(x_tsne, y_value, "t-sne")



if __name__ == '__main__':
    x_value, y_value = get_data()

    draw_simple_tsne_(x_value,y_value)