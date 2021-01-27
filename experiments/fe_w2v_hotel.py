import multiprocessing
import warnings

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn import preprocessing
import umap

from src.datasets import load_train_test


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    train_test = load_train_test()
    hotel_le = preprocessing.LabelEncoder()
    train_test['hotel_country'] = hotel_le.fit_transform(train_test['hotel_country'])
    train_test['hotel_country'] = train_test['hotel_country'].astype(str)

    train = train_test[train_test['row_num'].isnull()]
    test = train_test[~train_test['row_num'].isnull()]

    # Delete last hotel_country to avoid leakage
    train_hotels = train.groupby('utrip_id')['hotel_country'].apply(lambda x: x.values[:-1]).reset_index()
    test_hotels = test.groupby('utrip_id')['hotel_country'].apply(lambda x: x.values).reset_index()
    hotels = pd.concat([train_hotels, test_hotels], sort=False)

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=1,
                         window=5,
                         size=512,
                         alpha=0.03,
                         min_alpha=0.0007,
                         workers=cores - 1)

    sentences = [list(ar) for ar in hotels['hotel_country'].to_list()]
    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=25)
    w2v_model.save("word2vec.model")

    data = np.array([w2v_model.wv[d] for d in w2v_model.wv.vocab.keys()])
    print(data.shape)
    result = umap.UMAP(n_neighbors=5, n_components=2).fit(data)
    pd.DataFrame(result.embedding_).plot.scatter(x=0, y=1)
