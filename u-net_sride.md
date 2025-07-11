---
marp: true
---
<!--
headingDivider: 1
-->


# Abstract

U-Netの目的は医学的画像セグメンテーション。しかし、医学的画像は何千枚ものアノテーション付きデータを用意することが難しい。

そこでFCNアーキテクチャを修正・拡張し、少数の学習画像でも機能し、より精密なセグメンテーションを実現できるようにしたのがU-Net。


---

さらに、U-Netは高速かつ高精度。512×512の画像のセグメンテーションには、最近のGPUで1秒未満しかかからない。

部分的にアノテーションされた35枚のトレーニング画像から構成される「PhC-U373」データセット、部分的にアノテーションされた20枚のトレーニング画像から構成される「DIC-HeLa」データセット下記の表のような精度を記録。

![](./img/image%20copy%202.png)

表2. ISBI Cell Tracking Challenge 2015におけるセグメンテーション結果（IOU）。

---

![](./img/image%20copy.png)

図4. ISBI細胞追跡チャレンジにおける結果。
(a) 「PhC-U373」データセットの入力画像の一部。(b) セグメンテーション結果（シアンのマスク）と手動で作成された正解データ（黄色の枠線）。(c) 「DIC-HeLa」データセットの入力画像。(d) セグメンテーション結果（ランダムな色のマスク）と手動で作成された正解データ（黄色の枠線）。

# 1.Introduction


FCNアーキテクチャにおける重要な修正点の一つは、アップサンプリング部分においても多くの特徴チャネルを持たせていることです。

これにより、ネットワークはコンテキスト情報を高解像度の層へ伝播させることができます。

その結果、contextを捉えるための拡張経路（expansive path）、精密な位置特定を可能にする対称的な拡張パス収縮経路（contracting path）はほぼ対称になっており、U字型のアーキテクチャが得られます。

---

![](./img/image%20copy%203.png)

---



多くの細胞セグメンテーションタスクにおけるもう一つの課題は、同じクラスに属する接触しているオブジェクトの分離です（図3参照）。

![](./img/image%20copy%205.png)

図3. DIC（微分干渉コントラスト）顕微鏡を用いてガラス上で記録されたHeLa細胞。
(a) 生画像。(b) 正解セグメンテーションとの重ね合わせ。異なる色が異なるHeLa細胞のインスタンスを示している。(c) 生成されたセグメンテーションマスク（白：前景、黒：背景）。(d) ネットワークに境界ピクセルの学習を促すための、画素単位の損失重み付きマップ。

---

このタスクに対応するためにU-Netは **重み付き損失関数（weighted loss）** を使用します。

$$
E = \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log(p_{\ell(\mathbf{x})}(\mathbf{x}))
\tag{1}
$$

ここでは、接触している細胞の間を分離する背景ラベルに対して損失関数内で大きな重みを与えます。この際に使用される重みマップ $w$ は事前に計算され、上記の一番右のようなデータが例です。


# 2.Network Architecture


ネットワークアーキテクチャは下記のようになります。

---

![](./img/image%20copy%203.png)

---


これは、収縮経路（左側）と拡張経路（右側）から構成されています。

収縮経路は、畳み込みネットワークの典型的なアーキテクチャに従っている。

これは、3×3の畳み込み（パディングなし）を2回適用し、ReLU関数の適用と、ストライド2による2×2の最大プーリング操作を行うという繰り返しで構成されています。

各ダウンサンプリングステップ(最大プーリング)ごとに、特徴チャネルの数を倍にします。

---

拡張経路の各ステップでは、特徴マップのアップサンプリング、特徴チャネルの数を半分にする2×2の畳み込み（「デコンボリューション」）、収縮経路から対応するクロップ（＝サイズの調整、ex）中央部分の切り抜き）された特徴マップとの連結、そして2回の3×3の畳み込み（それぞれReLUの後に続く）を行う。

クロッピングが必要なのは、各畳み込みにおいてパディングを行っておらずデータのサイズが畳み込みのたびに小さくなるため、デコンボリューションされた拡張経路と、元の収縮経路のデータのサイズが違うから。

最終層では、1×1の畳み込みによって、各64成分の特徴ベクトルを目的のクラス数にマッピングします。ネットワーク全体で、畳み込み層は23層あります。

# 3.Training

U-Netにおける損失関数は、最終的な特徴マップに対して画素単位のソフトマックスを適用し、クロスエントロピー損失関数と組み合わせて計算されます。

ソフトマックスは以下のように定義されます：

$$
p_k(\mathbf{x}) = \frac{\exp(a_k(\mathbf{x}))}{\sum_{k'=1}^K \exp(a_{k'}(\mathbf{x}))}
$$

ここで、$a_k(\mathbf{x})$ は、位置 $\mathbf{x} \in \Omega$ における特徴チャネル $k$ の活性値を表します（$\Omega \subset \mathbb{Z}^2$）。


$K$ : クラスの数

---

最終的な損失関数は下記のようになります

$$
E = \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log(p_{\ell(\mathbf{x})}(\mathbf{x}))
\tag{1}
$$

ここで、$\ell : \Omega \rightarrow \{1, \dots, K\}$ は各画素の正解ラベルを表し、
$w : \Omega \rightarrow \mathbb{R}$ は、訓練時に特定の画素により重要性を与えるために導入された重みマップです。

---

各グラウンドトゥルースセグメンテーションに対して、重みマップ $w : \Omega \rightarrow \mathbb{R}$ を事前に計算する。

これは、訓練データセット内で特定のクラスに属する画素の頻度の違いを補正するため。（背景などは頻度が高く、全体の損失関数に対して背景部分の損失の割合が高くなる＝背景とそのほかの識別に重きをおいてしまう）

および接触している細胞間に導入した小さな分離境界をネットワークに学習させるためです（図 3c および d を参照）。

![](./img/image%20copy%205.png)

---

重みマップは次のように計算される：

$$
w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \exp\left( -\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2} \right)
\tag{2}
$$

ここで：

- $w_c : \Omega \to \mathbb{R}$ は、クラス頻度を補正するための重みマップ背景ピクセルと細胞ピクセルの出現頻度が異なる。そのため、頻度の低いクラスに対して大きな重みを与える設計になっている(ex. $\mathbf{x}$ のピクセルが背景なら0、それ以外なら1)
- $d_1 : \Omega \to \mathbb{R}$ は、最も近い細胞の境界までの距離
- $d_2 : \Omega \to \mathbb{R}$ は、2番目に近い細胞の境界までの距離

論文では、$w_0 = 10$ および $\sigma \approx 5$ ピクセルと設定

---

U-Netでは、一般的なCNNと同じ、初期重みをガウス分布（正規分布）から以下の標準偏差でサンプリングする手法を取る（=He初期化）

$$
\text{標準偏差} = \sqrt{\frac{2}{N}}
$$

ここで $N$ は、ある1つのニューロンに入力されるノード（ユニット）の数を表します[^5]。

例えば、3×3の畳み込みと、前の層に64チャネルの特徴マップがある場合には：

$$
N = 3 \cdot 3 \cdot 64 = 576
$$


