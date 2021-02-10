# load_text.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from wordcloud import WordCloud

class bbctext:

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent
    	# stop_word="english" removes common english words the, to, as
    	# set limits on doc frequency
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df = 0.7, min_df = 0.001)

    def load(self,nsample=2225):
        # read data from file
        df = pd.read_csv(self.root_dir / "Data_BBCText/bbc-text.csv")
        # select nsample documents and create feature matrix
        Xdf = df["text"][0:nsample]
        X_tfidf = self.vectorizer.fit_transform(Xdf).toarray().T
        # extract class label
        class_label = df['category'].values[0:nsample]
        print("X.shape: {} - labels.shape: {}".format(X_tfidf.shape, class_label.shape))
        return X_tfidf, class_label

    def create_wordcloud(self,X_tfidf,cluster_assignment,ncluster,nword=50):
        nrow = 2
        ncol = 3
        fig, ax = plt.subplots(nrow,ncol)
        ax[1,2].axis("off")
        for cluster in range(ncluster):
            row = int(cluster/ncol)
            col = cluster % ncol
            idx = np.where(np.absolute(cluster_assignment-cluster)<1e-5)[0]
            influence = np.sum(X_tfidf[:,idx],axis=1)
            idx_influence_most = np.argsort(-influence)[0:nword]
            word_influence_most = np.array(self.vectorizer.get_feature_names())[idx_influence_most]
            word_dict = {word_influence_most[i]:influence[idx_influence_most[i]] for i in range(nword)}
            print("Cluster: {}  \nwords: {}".format(cluster,word_influence_most[0:10]))
            wc = WordCloud(background_color="white",width=1000,height=600, max_words=100,relative_scaling=0.5).generate_from_frequencies(word_dict)
            ax[row,col].imshow(wc)
            ax[row,col].axis("off")
            ax[row,col].set_title("Cluster "+str(cluster))
        plt.show()

if __name__ == "__main__":
	text = bbctext()
	X,Y = text.load()