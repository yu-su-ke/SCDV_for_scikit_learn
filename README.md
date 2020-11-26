# SCDV for scikit-learn
[SCDV : Sparse Composite Document Vectors using soft clustering over distributional representations](https://www.aclweb.org/anthology/D17-1069/)  
を参考にscikit-learnのPipelineなどに使えるようにTransformerMixinとBaseEstimatorを継承したクラスとして再実装したものである。  


## 環境構築
### 1. Install library
Python >= 3.7.0 でその他ライブラリのインストールは以下のコマンドを実行する。  
```shell script
$ pip install -r requirements.txt
```


### 2. How to use
任意のpythonファイル内で`SCDV.py`を読み込む。
```python
from SCDV_for_scikit_learn.SCDV import SparseCompositeDocumentVector
```
分類器と合わせて、Pipelineに入れる。(以下はRandom Forestの場合の例)  
```python
pipe = Pipeline([('scdv', SparseCompositeDocumentVector()),
                 ('rfc', RandomForestClassifier())])
param_grid = {
    'scdv__num_clusters': [i for i in range(20, 100, 10)],
    'scdv__num_features': [100, 200, 300],
    'scdv__min_word_count': [0, 10, 20],
    "rfc__n_estimators": [i for i in range(1, 21, 2)],
    "rfc__criterion": ["gini", "entropy"],
    "rfc__max_depth": [None, 1, 2, 3, 4],
    "rfc__random_state": [i for i in range(0, 101, 20)],
}
```
後はその他の設定をし、学習を始める。  
学習中に構築されたword2vecモデルや各単語のクラスタ割り当て確率を記録した辞書などは`model`ディレクトリに保存される。  
パラメータの1つである`save_directory`で任意のディレクトリを指定可能。  


### 3. Extra
SCDV用のパラメータは以下の通り、
```
num_features  # 単語次元数
min_word_count  # 最小カウントワード
num_workers  # 並行処理するスレッドの数
context  # コンテキストウィンドウサイズ
down_sampling  # 頻出単語のダウンサンプル設定
num_clusters  # クラスタ数
percentage  # スパースにするためのしきい値パーセンテージを設定
save_directory     # 各種モデルを保存するディレクトリ
```


## 参考文献
[1] D. Mekala, V. Gupta, B. Paranjape, and H. Karnick, "SCDV: Sparse Composite Document Vectors using soft clustering over distributional representations," 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 659–669, 2017, doi: 10.18653/v1/d17-1069.
[2] D. Mekala, V. Gupta, B. Paranjape, "SCDV", Github, https://github.com/dheeraj7596/SCD .