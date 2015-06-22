# Bulding Machine Learning Systems with Python
# 実践 機械学習

## 7章 回帰:レコメンド
- 最小二乗法による回帰(古典的手法)の復習
	- 高速に実行できる
	- 現実の多くの問題に対して効果的
	- 特徴量の数がサンプルデータの数より大きい場合正しく扱えない
- Numpy, scikit-learnによる実行
- より進んだ手法 (lasso、ridge、elastic nets) の紹介
	- 古典的手法でうまく扱えない問題(特徴量の数が多い場合など)に対して有効
- レコメンドについての基礎

###7.1 回帰を用いて物件価格を予測する
ボストンにある物件について、その価格を予測する問題を考える。

#### 目標
    入力 : エリアの指定
    出力 : その周辺にある家の値段の中央値 (median)

#### データセット
以下の情報がデータセットで与えられる

- 犯罪発生率
- 人口統計に関する情報
	- 「先生一人あたりに対する生徒の数」など 
- 地理情報

#### データセットの読み込み
scikit-learnに組み込まれているため以下のようにデータを読み込める

```python
from sklearn.datasets import load_boston
boston = load_boston()
```

`boston` オブジェクトは属性 (attribute) をいくつか持ち 、主に以下を用いる
- `boston.data` : 特徴量がいろいろ入っている
- `boston.target` : ラベル、各エリアの家の値段の中央値 (今回の問題の目標変数 : target variable)
     
`boston. DESCR`, `boston.feature_names` でデータセットの詳細が確認できる。
