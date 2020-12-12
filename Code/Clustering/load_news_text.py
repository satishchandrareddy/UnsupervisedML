# load_text_news.py

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# subset can be 'train' or 'test'
def load_news_text(subset='train'):
	raw_posts = fetch_20newsgroups(subset=subset, shuffle=True, random_state=99)
	class_mapper = {'rec.sport.hockey': 'Sports', 
                'rec.sport.baseball': 'Sports',
                'talk.religion.misc': 'Religion',
                'talk.religion.misc': 'Religion',
                'alt.atheism': 'Religion',
                'soc.religion.christian': 'Religion',
                'sci.electronics': 'Technology',
                'comp.graphics': 'Technology',
                'comp.sys.ibm.pc.hardware': 'Technology',
                'comp.os.ms-windows.misc': 'Technology',
                'comp.windows.x': 'Technology'
	}
	labels = pd.Series(raw_posts.target).map(lambda x: raw_posts.target_names[x])
	df = pd.DataFrame({'label': labels, 'text': raw_posts.data})
	df['label'] = df['label'].map(class_mapper)
	df.dropna(inplace=True)
	vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=0.001)
	X_tfidf = vectorizer.fit_transform(df['text']).toarray().T
	labels = df['label'].values
	print("X.shape: {} - labels.shape: {}".format(X_tfidf.shape, labels.shape))
	
	return X_tfidf, labels
	