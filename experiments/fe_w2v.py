import multiprocessing

from gensim.models import Word2Vec
from sklearn import preprocessing

from src.datasets import load_train_test


if __name__ == '__main__':
    train_test = load_train_test()
    target_le = preprocessing.LabelEncoder()
    train_test['city_id'] = target_le.fit_transform(train_test['city_id'])
    train_test['city_id'] = train_test['city_id'].astype(str)

    train = train_test[train_test['row_num'].isnull()]
    train_trips = train[train['city_id'] != train['city_id'].shift(1)].groupby('utrip_id')['city_id'].apply(lambda x: x.values).reset_index()

    train_trips['n_trips'] = train_trips['city_id'].map(lambda x: len(x))

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=1,
                         window=4,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1)

    sentences = [list(ar) for ar in train_trips['city_id'].to_list()]
    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=1)
    w2v_model.save("word2vec.model")
