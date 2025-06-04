---
marp: true
---
<!--
headingDivider: 1
-->




# Fully Convolutional Networks for Semantic Segmentation




Fully Convolutional Networks：全結合層を持たず、線形層が全て畳み込み層だけで構成されているCNN。そのため、Deconvolution（転置畳み込み）を用いることで任意のサイズの入力を受け取り、対応するサイズの出力を効率的に推論・学習することが可能なネットワーク

---

## Deconvolution(転置畳み込み) = 畳み込みの逆操作

![](./img/image%20copy%208.png)

入力データを高解像度にすることが出来る。

---

### FCNでないモデル(ex.VGG16)

![](./img/image%20copy%207.png)

---

本論文ではFCNをセグメンテーションに応用する手法を提案します。


CNNは画像分類など、画像全体に対する推論（＝粗い推論）は2014年当時成功していた。

そのため次のステップとして、あらゆるピクセルに対して予測を行う(ex.セマンティックセグメンテーション)研究が進められていた。

本論文での提案手法登場以前も、ConvNet をセマンティックセグメンテーションに使用してきたが、精度・推論時間などの欠点があり、それを解決したのが本論文の提案手法である。


---


## 最初に提案されたアーキテクチャ = Encoder部分とDecoder部分からなるFCNアーキテクチャ


![](./img/image%20copy%209.png)


---


Encocer部分：従来のCNNに似ており、畳み込み層とプーリング層によって構成され、空間的な次元を段階的に縮小しつつ、特徴チャネルの数を増加させます。


Decoder部分：この部分では、転置畳み込み(UpSampling)を用いて特徴マップを元の入力解像度までアップサンプリングします。最終出力層にsoftmaxを適用し、各クラスの確率を推定している。


また、セグメンテーションタスクにおけるFCNの損失関数はピクセルごとのクロスエントロピー損失（pixel-wise cross-entropy loss）の総和になる。

---

入力画像 $\boldsymbol{x} \in \mathbb{R}^{H \times W \times C}$ に対して、モデル出力を

$$
\hat{\boldsymbol{y}}_{ij} = \mathrm{softmax}(\boldsymbol{z}_{ij}) \in \mathbb{R}^K
$$

とします。

ここで：

- $\boldsymbol{z}_{ij}$ は画素 $(i, j)$ における各クラスのスコア（logits）
- $K$ はクラス数
- $\hat{\boldsymbol{y}}_{ij,k}$ は画素 $(i, j)$ がクラス $k$ である確率

正解ラベル $\boldsymbol{y}_{ij} \in \{0, 1\}^K$（ワンホットベクトル）に対して、損失関数 $\mathcal{L}$ は次のように定義されます：

$$
\mathcal{L} = -\sum_{i=1}^H \sum_{j=1}^W \sum_{k=1}^K y_{ij,k} \log \hat{y}_{ij,k}
$$
---


学習済み画像分類モデルへの適用


画像分類タスクで学習されたAlexNet、VGG net、および GoogLeNetのネットワークに変更を加えることでFCNにし、それらの学習済み表現をファインチューニングによってセグメンテーションタスクに転用することができる。


---


ex）VGG16




![](https://private-user-images.githubusercontent.com/82156802/450592708-7dc2f454-ebb5-48b9-8662-9e554a065400.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDg5MzIyMzgsIm5iZiI6MTc0ODkzMTkzOCwicGF0aCI6Ii84MjE1NjgwMi80NTA1OTI3MDgtN2RjMmY0NTQtZWJiNS00OGI5LTg2NjItOWU1NTRhMDY1NDAwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjAzVDA2MjUzOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjY2EyMGNkYTc5MDE4ZjM1ZWY5OGM0MTUwMTJiODE5YjdhYmJmODEwNmYxYTM4MmUyMzBlOGM4YjJlM2Q1NmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.KuVqN_aEqxgpLlGDrgdFbcAO3tFXPNvWKJxGSXjhOC0)






最終出力層を取り除く。


そして、最後の２つの全結合層をConvolutionレイヤーと捉えることで、Deconvolutionをモデルの最後に追加することができる。


---




## スキップ接続を用いたアーキテクチャ(これがU-Netにつながった)




ストライド付き畳み込み（strided convolution）やプーリングで解像度を下げると，ピクセル単位の細かな位置情報は捨てられる。


セグメンテーションにおいて、位置情報が失われてしまうと精度は出ない...


そこで、スキップ接続というものを上記のFCNアーキテクチャに追加することが論文で提案された。


スキップ接続により、上層のまだ位置情報を次元的に圧縮仕切っていない情報を、下層のアップサンプリングの際に足し合わせることで精度の良いセグメンテーションを実現する。


---


![Image](https://private-user-images.githubusercontent.com/82156802/450591244-69c8dee9-d962-4c25-bf02-588769299683.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDg5MzE5OTcsIm5iZiI6MTc0ODkzMTY5NywicGF0aCI6Ii84MjE1NjgwMi80NTA1OTEyNDQtNjljOGRlZTktZDk2Mi00YzI1LWJmMDItNTg4NzY5Mjk5NjgzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA2MDMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNjAzVDA2MjEzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY3ODIzYzk5YzRkNGFhYjJlNmI1ZDQ0YzVhYzZiYTgyMWI0YmI5NTU2Y2MxODM1MDI0MzA4ZDU2MDdhNDEzN2QmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.7G6_mZGvZd9_6QCXQ3vItAkt1Lu2neDVCV7nEca3BoU)




---


スキップ接続にも、
