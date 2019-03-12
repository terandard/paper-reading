Hu H, Chao W, Fei S  
CVPR2018  
[arXiv](https://arxiv.org/abs/1806.03724), [pdf](https://arxiv.org/pdf/1806.03724.pdf), [GitHub](https://github.com/hexiang-hu/answer_embedding)  

# どんなもの？
新しい確率モデル (Visual QA) を提案  
$(i,q)$ の埋め込みだけでなく，$a$ の埋め込みも学習する．  
転移学習に効果的である．

# 先行研究との差分
オープンエンドのVQAでは各 $(i,q)$ に対して K-way classifier を構築するのが一般的  
しかし，2つの異なる答え $a_k$ と $a_l$ をクラス $k$ と $l$ とみなすので，意味的な関連性を失う．  
加えて，候補回答集合 $A$ が重複しないような2つのデータセットにまたがって一般化するのには適していない．

提案手法では $(i,q)$ の埋め込みだけでなく，$a$ の埋め込みも学習する．埋め込み関数は適切に学習されると，類似性を維持し，（トレーニングデータ内の）見えない答えに一般化する．


# abstract
学習目的は、正解が全ての候補回答集合の中でより高い確率を持つように、それらの埋め込みの最良のパラメータ化を学ぶことです。多元分類としてビジュアルQAを扱ういくつかの既存のアプローチとは対照的に、提案されたアプローチは、独立した序数としてそれらを見るのではなく、答えの間の意味的関係（埋め込みによって特徴付けられる）を考慮に入れる。したがって、学習された埋め込み関数を使用して、目に見えない答えを（トレーニングデータセットに）埋め込むことができます。これらの特性は、モデルが学習されるソースデータセットが解答空間内のターゲットデータセットとの重複を制限しているオープンエンドのビジュアルQAの転移学習に特に魅力的なアプローチになります。また、提案された確率モデルを正しく正規化することが課題である、多数の回答を持つデータセットにモデルを適用するための大規模最適化手法も開発しました。


# introduction
私たちの主なアイデアは、答えの埋め込みも学ぶことです。 いくつかの空間における画像および質問の（共同埋め込み）特徴と一緒に、回答埋め込みは、回答が画像と質問のペアにどのように似ているかを記述する確率モデルをパラメータ化します。 正解の可能性を最大にするために、答えと画像や質問への埋め込みを学びます。 学習されたモデルは、このように答えの意味的類似性を画像と質問の対の視覚的/意味的類似性と整列させる。 さらに、学習されたモデルは目に見えない答えを埋め込むこともできるため、あるデータセットから別のデータセットに一般化することができます。

私たちの方法は何百、何千もの答えの埋め込みを学ぶ必要があります。 したがって、確率モデルを最適化するために、ミニバッチで負の例を適応的にサンプリングするという計算効率の良い方法を導入することで、この課題を克服します。

我々のモデルはまた、画像と質問の各ペアに対して、どれだけの候補回答を調べなければならないかに関係なく、画像と質問の同時埋め込みを一度計算するだけでよいという計算上の利点を有する。 反対に、[13、7]のようなモデルはトリプレット（画像、質問と回答）の共同埋め込みを学習し、候補回答数の線形オーダーで埋め込みを計算する必要があります。 回答候補数を多くする必要がある場合（カバレッジを向上させるため）、そのようなモデルは簡単にスケールアップできません。


# model

回答集合 $A$ を $C$ と $D = A-C$ に分ける  
$C$ : 全ての正解回答(類義語を含む ex: “policeman” , “police officer”)  
$D$ : 誤り(もしくは望ましくない)回答  

訓練データセットは、Ｎ個の特有のトリプレットの集合で以下の様に表される :  
- 正解のみが与えられたときには $D=\{(i_n,q_n,C_n)\}$ ,  
- 正解と不正解の両方が与えられたときには $D=\{(i_n,q_n,A_n=C_n\cup D_n)\}$  

### embedding
$f_\theta(i,q)$ : $(i,q)$ の embedding  
$g_\phi(a)$ : $a$ の embedding   
multi-layer perceptron (MLP [[13](https://arxiv.org/abs/1606.08390)], [[5](https://arxiv.org/abs/1704.07121)]) や Stacked Attention Network（SAN [[30](https://arxiv.org/abs/1511.02274)],[[15](https://arxiv.org/abs/1704.03162)]）を使用

### Probabilistic Model of Compatibility (PMC)

$a$ が正解の回答である $(i_n,q_n,a\in C_n)$ が得られると，次の確率モデルを定義する:

```math
p(a|i_n,q_n) = \frac{\exp(f_\theta(i_n,q_n)^Tg_\phi(a))}{\sum_{a'\in A}\exp(f_\theta(i_n,q_n)^Tg_\phi(a'))}
```

### Discriminative Learning with Weighted Likelihood
次のような重み付き尤度がより効果的であることが分かった．

```math
l = -\sum_n^N\sum_{a\in C_n}\sum_{d\in A}\alpha(a,d)\log P(d|i_n,q_n)
```

```math
\alpha(a,d)=\mathbb{I}[a=d]
```

$\mathbb{I}[\cdot]$ はバイナリインジケーター(1 : true, 0 : false)  
$C_n$ がシングルトンの場合，目的関数は標準のクロスエントロピー損失まで減少


## Large-scale Stochastic Optimization
重み付き尤度を最適化するために，ミニバッチベースの確率勾配降下法を使用．$D$ から $B$ トリプレットをランダムに選択し，重み付き尤度の勾配を計算．

ミニバッチ $b=1,2,...,B$ 内の $(i_b,q_b,C_b)$ もしくは $(i_b,q_b,C_b\cup D_b)$  について  

```math
A_B=\cup_{b=1}^N(C_b\cup D_b)
```

ミニバッチのすべての可能な答えを使用  
加えて負のサンプリングを使用して補強する．

```math
\bar{A}_B=A-A_B
```

このセットから $M$ サンプル取得する．これらのサンプルを $A_o$ とし，$A_B$ と合わせる．つまり，$p(a|i,q)$ と尤度を計算する際に，$A$ の代わりに $A_o\cup A_B$ を使用する．


## Defining the Weighting Function
重み関数を利用して、外部または以前の意味論的知識を組み込むことができます。  
例えば、$\alpha(a,d)$ は，$a$ と $d$ との間の意味的類似度スコアに依存し得る。

WUPS を使用し，次のルールを定義する  

```math
\alpha(a,d) = \left\{
\begin{array}{ll}
1 & if \textbf{WUPS}(a,d) > \lambda \\
0 & \text{otherwise}
\end{array}
\right.
```

$\alpha(a,d)$ は、$C$ の意味的に類似した多くの答えでトリプレットをスケーリングするためにも使用できます

```math
\alpha(a,d)=\frac{\mathbb{I}[a=d]}{|C|}
```

これらの同様の答えはそれぞれ、目的関数に対する可能性のごく一部にしか寄与しないようにします。 eq（7）のアイデアは、VQA [3]とVQA2 [9]の性能を向上させるための最近のいくつかの研究[32、12、15]で利用されています。

## prediction
トレーニング中，以下の decision rule を適応する

```math
a^*=\text{arg max}_{a\in A}f_\theta(i,q)^Tg_\phi(a)
```

ここで $g_\phi$ を使用することによって柔軟性が得られる． 


## Comparison to Existing Algorithms
ほとんどの既存のVQAアルゴリズムは $f_\theta$ に加えて multi-way classifier を訓練する．

三変数互換性(tri-variable compatibility)関数 $h(i,q,a)$ を学習するアルゴリズムもある[13,7,25]． そして正しい回答は，$h(i,q,a^*）$ が最高になるように $a^*$ を識別することによって推論される．(後の uPMC)

提案した decision rule は $h(i,q,a)$ の因数分解形式である $f_\theta(i,q)^Tg_\phi(a)$ の計算に依存している．実際にはこの因数分解のために，各ペア $(i,q)$ について一度だけ $f_\theta(i,q)$ を計算するだけでよい． $g_\phi(a)$ については，モデルが十分に単純である限り，可能な多くの $a$ を列挙することは，関数 $h(i,q,a)$ が必要とするものよりも要求が厳しくない．実際には，任意の可能な $a$ に対して $g_\phi(a)$ を一度計算するだけでよい．(後の fPMC)


# 評価実験

データセット  
![table1]  
![table2]  

$f_\theta$ をパラメータ化するために，MLP と SAN を使用する．  

## MLP
![fig4]  

## SAN
![fig5]  

$g_\phi$ をパラメータ化するために，2つのアーキテクチャを使用する．

 1) Utilizing a one-layer MLP on average GloVe embeddings of answer sequences, with the output dimensionality of 1,024.   
 2) Utilizing a two-layer bidirectional LSTM (bi-LSTM) on top of GloVE embeddings of answer sequences. 


![table3]  

ネガティブサンプリングの効果  
![table4]  
![fig2]  
少しのサンプルを加えるだけでも精度が良くなる．



データセット間で共通の回答数  
![table6]  


(行) から (列) に転移した結果
![table5]  
fPMC はすべての転移設定において uPMC よりも優れている  
VQA2に yes/no の回答が多数含まれているため，V7W や qaVG から転移するのが難しい．

yes/no の回答を除いた場合  
![table7]  


fPMC,uPMC,CLSの推論効率．すべてのアプローチに対して1つの隠れ層MLPモデルを使用し，$|C| = 1000$ を維持し，ミニバッチサイズを128とする．VQA2 validation で評価  
![table8]  
![fig3]


answer embedding の可視化  
![fig6]

