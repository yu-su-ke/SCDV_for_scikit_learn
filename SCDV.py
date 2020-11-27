import joblib
import logging
import os

from gensim.models import Word2Vec
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseCompositeDocumentVector(TransformerMixin, BaseEstimator):
    def __init__(self, num_features=200, min_word_count=20, num_workers=os.cpu_count(), context=10,
                 down_sampling=1e-3, num_clusters=60, percentage=0.04, save_directory='./SCDV_for_scikit_learn/model'):
        self.num_features = num_features  # 単語次元数
        self.min_word_count = min_word_count  # 最小カウントワード
        self.num_workers = num_workers  # 並行処理するスレッドの数
        self.context = context  # コンテキストウィンドウサイズ
        self.down_sampling = down_sampling  # 頻出単語のダウンサンプル設定
        self.num_clusters = num_clusters  # クラスタ数
        self.percentage = percentage  # スパースにするためのしきい値パーセンテージを設定
        self.save_directory = save_directory  # 各種モデルを保存するディレクトリ

        self.variable_dictionary = {}

    def fit(self, X, y=None):
        """ 前処理で必要な学習の実施。yが存在しない場合もy=Noneで与えておく

        Args:
            X (list): 分かち書きしたテキストリスト
            y (list): ラベルリスト

        Returns:
            variable_dictionary (dict): 学習に必要な各変数を辞書として記録
                prob_word2vec (dict): 確率で重み付けしたword2vec
                word_centroid_dictionary (dict): 単語に割り当てられたクラスタの辞書
                word_centroid_prob_dictionary (dict): 単語に割り当てられたクラスタの確率をマップした辞書
                word_idf_dictionary (dict): 単語に割り当てられたidf値をマップした辞書
                feature_names(list): 単語リスト

        """
        word2vec_model = self.create_word2vec(X)
        # 語彙内のすべての単語のワードベクトルを取得
        word_vectors = word2vec_model.wv.syn0
        idx, idx_prob = self.cluster_gmm(word_vectors)
        # 単語/索引辞書を作成し、各語彙単語をクラスター番号にマッピング
        word_centroid_dictionary = dict(zip(word2vec_model.wv.index2word, idx))
        # 単語/クラスタ割り当ての確率辞書を作成し、各語彙単語をクラスタ割り当ての確率のリストにマッピング
        word_centroid_prob_dictionary = dict(zip(word2vec_model.wv.index2word, idx_prob))
        feature_names, word_idf_dictionary = self.tf_idf(X)
        # 確率単語語クラスターベクトルの事前計算
        prob_word_vectors = self.get_probability_word_vectors(feature_names, word_centroid_dictionary,
                                                              word_idf_dictionary,
                                                              word_centroid_prob_dictionary, word2vec_model)
        # SCDVのモデル
        self.variable_dictionary = {
            'prob_word2vec': prob_word_vectors,
            'word_centroid_dictionary': word_centroid_prob_dictionary,
            'word_centroid_prob_dictionary': word_centroid_prob_dictionary,
            'word_idf_dictionary': word_idf_dictionary,
            'feature_names': feature_names
        }
        joblib.dump(self.variable_dictionary, os.path.join(self.save_directory, 'variable.jb'), compress=3)
        return self

    def transform(self, X, train_flag=True, train_variable=None):
        """ 引数Xに前処理を適用する

        Args:
            X (list): 分かち書きしたテキストリスト
            train_flag (bool): 学習時と推論時の判定フラグ

        Returns:
            numpy.ndarray: 文書のSCDV

        """
        # 前処理を適用する際に、学習しておくべきパラメータがあるか確認
        check_is_fitted(self, 'variable_dictionary')
        # sklearnの入力の検証
        # X = check_array(X, accept_sparse=True)
        
        # テスト時は、変数群を引数から受け取る
        if train_flag is False:
            self.variable_dictionary = train_variable

        gwbowv = self.scdv(X, train_flag)
        return gwbowv

    def fit_transform(self, X, y=None, **fit_params):
        """

        Args:
            X (list): 分かち書きしたテキストリスト
            y (list): ラベルリスト
            **fit_params:

        Returns:
            numpy.ndarray: 文書のSCDV

        """
        word2vec_model = self.create_word2vec(X)

        # 語彙内のすべての単語のワードベクトルを取得
        word_vectors = word2vec_model.wv.syn0
        idx, idx_prob = self.cluster_gmm(word_vectors)
        # 単語/索引辞書を作成し、各語彙単語をクラスター番号にマッピング
        word_centroid_dictionary = dict(zip(word2vec_model.wv.index2word, idx))
        # 単語/クラスタ割り当ての確率辞書を作成し、各語彙単語をクラスタ割り当ての確率のリストにマッピング
        word_centroid_prob_dictionary = dict(zip(word2vec_model.wv.index2word, idx_prob))
        feature_names, word_idf_dictionary = self.tf_idf(X)
        # 確率単語語クラスターベクトルの事前計算
        prob_word_vectors = self.get_probability_word_vectors(feature_names, word_centroid_dictionary,
                                                              word_idf_dictionary, word_centroid_prob_dictionary,
                                                              word2vec_model)
        # SCDVのモデル
        self.variable_dictionary = {
            'prob_word2vec': prob_word_vectors,
            'word_centroid_dictionary': word_centroid_prob_dictionary,
            'word_centroid_prob_dictionary': word_centroid_prob_dictionary,
            'word_idf_dictionary': word_idf_dictionary,
            'feature_names': feature_names
        }
        joblib.dump(self.variable_dictionary, os.path.join(self.save_directory, 'variable.jb'), compress=3)
        gwbowv = self.scdv(X)
        return gwbowv

    def create_word2vec(self, X):
        """ word2vecの学習

        Args:
            X (list): テキストリスト

        Returns:
            gensim.models.word2vec.Word2Vec: 学習データで構築したword2vecモデル

        """
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        word2vec_model = Word2Vec(X, workers=self.num_workers, hs=0, sg=1, negative=10, iter=25, size=self.num_features,
                                  min_count=self.min_word_count, window=self.context, sample=self.down_sampling, seed=1)
        word2vec_model.init_sims(replace=True)
        # Word2Vecモデルを保存
        logger.info('Saving Word2Vec model...')
        model_name = '{}features_{}minwords_{}context_len2alldata'.format(
            self.num_features, self.min_word_count, self.context)
        word2vec_model.save(os.path.join(self.save_directory, '{}.model'.format(model_name)))
        return word2vec_model

    def cluster_gmm(self, word_vectors):
        """ GaussianMixtureを使って、クラスタリングを構築

        Args:
            word_vectors (numpy.ndarray): 語彙内のすべての単語のワードベクトル

        Returns:
            numpy.ndarray: 割り当てられたクラスタ
            numpy.ndarray: 各単語のクラスタ割り当て確率

        """
        clf = GaussianMixture(n_components=self.num_clusters, covariance_type='tied', init_params='kmeans', max_iter=50)
        clf.fit(word_vectors)
        idx = clf.predict(word_vectors)
        idx_prob = clf.predict_proba(word_vectors)
        # 割り当てられたクラスタとその確率を保存する
        pickle.dump(idx, open(os.path.join(self.save_directory, 'gmm_latestclusmodel_len2alldata.pkl'), 'wb'))
        logger.info('Cluster Assignments Saved...')
        pickle.dump(idx_prob, open(os.path.join(self.save_directory, 'gmm_prob_latestclusmodel_len2alldata.pkl'), 'wb'))
        logger.info('Probabilities of Cluster Assignments Saved...')
        return idx, idx_prob

    def tf_idf(self, X):
        """

        Args:
            X (list): テキストリスト

        Returns:
            list: 単語リスト
            dict: 単語リスト内の各単語のidf値の辞書

        """
        train_data = [' '.join(i) for i in X]
        tfv = TfidfVectorizer(dtype=np.float32)
        tfidf_matrix_train_data = tfv.fit_transform(train_data)
        feature_names = tfv.get_feature_names()
        idf = tfv._tfidf.idf_

        # idf値にマップされた単語を使用して辞書を作成する
        logger.info('Creating word-idf dictionary for Training set...')
        word_idf_dict = {}
        for pair in zip(feature_names, idf):
            word_idf_dict[pair[0]] = pair[1]
        return feature_names, word_idf_dict

    def get_probability_word_vectors(self, feature_names, word_centroid_dictionary, word_idf_dictionary,
                                     word_centroid_prob_dictionary, word2vec_model):
        """ 確率で重み付けしたワードクラスタベクトルを計算する

        Args:
            feature_names (list): 単語リスト
            word_centroid_dictionary (dict): 単語に割り当てられたクラスタの辞書
            word_idf_dictionary (dict): 単語に割り当てられたidf値をマップした辞書
            word_centroid_prob_dictionary (dict): 単語に割り当てられたクラスタの確率をマップした辞書
            word2vec_model (gensim.models.word2vec.Word2Vec): 学習データで構築したword2vecモデル

        Returns:
            dict: 確率で重み付けしたワードベクトル

        """
        prob_word_vectors = {}
        for word in word_centroid_dictionary:
            prob_word_vectors[word] = np.zeros(self.num_clusters * self.num_features, dtype="float32")
            for index in range(0, self.num_clusters):
                try:
                    prob_word_vectors[word][index * self.num_features:(index + 1) * self.num_features] = \
                        word2vec_model[word] * word_centroid_prob_dictionary[word][index] * word_idf_dictionary[word]
                except:
                    continue
        return prob_word_vectors

    def scdv(self, X, train_flag=True):
        """

        Args:
            X (list): テキストリスト
            train_flag (bool): 学習時と推論時の判定フラグ

        Returns:
            numpy.ndarray: 文書のSCDV

        """
        gwbowv = np.zeros((len(X), self.num_clusters * self.num_features), dtype='float32')

        min_no = 0
        max_no = 0
        counter = 0
        for wakati_words in X:
            gwbowv[counter], min_no, max_no = self.create_cluster_vector_and_gwbowv(wakati_words, min_no, max_no)
            counter += 1
            if counter % 1000 == 0:
                logger.info('text Covered : ', counter)

        logger.info('Making sparse...')
        if train_flag is True:
            min_no = min_no * 1.0 / len(X)
            max_no = max_no * 1.0 / len(X)
            logger.info('Average min: ', min_no)
            logger.info('Average max: ', max_no)
        thres = (abs(max_no) + abs(min_no)) / 2
        thres = thres * self.percentage
        # 閾値未満の行列の値をゼロにする
        temp = abs(gwbowv) < thres
        gwbowv[temp] = 0

        # gwbowvの保存
        if train_flag is True:
            gwbowv_name = 'SDV_' + str(self.num_clusters) + 'cluster_' + str(self.num_features) + \
                          'feature_matrix_gmm_sparse.npy'
            np.save(os.path.join(self.save_directory, gwbowv_name), gwbowv)
        else:
            test_gwbowv_name = 'Test_SDV_' + str(self.num_clusters) + 'cluster_' + str(self.num_features) + \
                               'feature_matrix_gmm_sparse.npy'
            np.save(os.path.join(self.save_directory, test_gwbowv_name), gwbowv)

        return gwbowv

    def create_cluster_vector_and_gwbowv(self, wakati_words, min_no, max_no):
        """ SDV特徴ベクトルを計算する

        Args:
            wakati_words (list): 分かち書きした単語リスト
            min_no: 特徴ベクトルをスパースにするための閾値に用いる最小値
            max_no: 特徴ベクトルをスパースにするための閾値に用いる最大値

        Returns:

        """
        prob_word2vec = self.variable_dictionary['prob_word2vec']
        word_centroid_dictionary = self.variable_dictionary['word_centroid_prob_dictionary']
        word_centroid_prob_dictionary = self.variable_dictionary['word_centroid_prob_dictionary']
        word_idf_dictionary = self.variable_dictionary['word_idf_dictionary']
        feature_names = self.variable_dictionary['feature_names']
        bag_of_centroids = np.zeros(self.num_clusters * self.num_features, dtype='float32')
        for word in wakati_words:
            try:
                temp = word_centroid_dictionary[word]
            except:
                continue
            bag_of_centroids += prob_word2vec[word]
        norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
        if norm != 0:
            bag_of_centroids /= norm
        # 特徴ベクトルをスパースにするために、最小値と最大値を記録
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)
        return bag_of_centroids, min_no, max_no
