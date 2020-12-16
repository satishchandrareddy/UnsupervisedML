# load_news_text.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# subset can be 'train' or 'test'
def load_news_text(subset='train'):
	# read data from file
	root_dir = Path(__file__).resolve().parent.parent
	df = pd.read_csv(root_dir / "Data_News_Text/news_text.csv")

	vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=0.001)
	X_tfidf = vectorizer.fit_transform(df['text']).toarray().T
	labels = df['label'].values
	print("X.shape: {} - labels.shape: {}".format(X_tfidf.shape, labels.shape))
	
	return X_tfidf, labels
	