import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_fer_data(data_path="./data_embed_npy.npy",
                 label_path="./label_npu.npy"):
    
    data = np.load(data_path)
    label = np.load(label_path)
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features

color_map = ['#FF0000', '#0000FF', '#008000', '#FFFF00', '#800080', '#FFA500', '#00FFFF', '#FFC0CB', '#A52A2A', '#808080'] # 10个类，准备7种颜色

def plot_embedding_2D(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1],marker='o',markersize=1,color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_fer_data() 

    print('Begining......') 	

	# 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) 
    result_2D = tsne_2D.fit_transform(data)
    
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label, 't-SNE')	# 将二维数据用plt绘制出来
    fig1.show()
    plt.savefig("./T-SNE.png")
    plt.pause(50)
    
if __name__ == '__main__':
    main()
