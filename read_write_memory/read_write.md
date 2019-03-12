Seil Na, Sangho Lee, Jisung Kim, Gunhee Kim  
[arXiv](https://arxiv.org/abs/1709.09345) , [pdf](https://arxiv.org/pdf/1709.09345.pdf) , [github](https://github.com/seilna/RWMN)

# どんなもの
RWMN（ReadWrite Memory Network）という新しいメモリネットワークモデルを提案  
メモリの読み書き操作に高い容量と柔軟性を持たせることができる複数の畳み込み層で構成された read/write ネットワークを設計．  
[MovieQAベンチマーク](http://movieqa.cs.toronto.edu/home/) の複数のタスクで最高の精度を達成．  
ストーリーの内容だけでなく，キャラクターとその行動の理由などのより抽象的な情報をより理解できる可能性を示している．  
![fig1]

# 従来手法との違い
既存のメモリ拡張型ネットワークモデルは，各メモリスロットを独立したブロックとして扱う．  
RWMNでは多層CNNを使用することにより，シーケンシャルメモリセルをチャンクとして読み書きすることができる．隣接するメモリブロックが強い相関関係を持つことが多いため，シーケンシャルストーリーを表現するのに適している．



# 提案モデル
![fig2]  
RWMNは、適切な表現を有する映画コンテンツをメモリに格納し、所定のクエリに応答してメモリセルから関連情報を抽出し、5つの選択肢から正解を選択するように訓練される。  
・モデルの入力：  
（i）映画全体のビデオセグメントおよびサブタイトルのペア $S_{movie} = \{(v_1,s_1),...,(v_n,s_n)\}$ （$n_{ave}=1558$）  
（ii）映画の質問 $q$  
（iii）5つの回答候補 $a = \{a_1,...,a_5\}$   
出力は5つの回答候補に対する信頼スコアベクトル．

## (a) Movie Embedding
各サブショット $v_i$ とテキストセンテンス $s_i$ を以下のように特徴表現に変換する．  
各フレーム $v_{ij}\in v_i$ について，ImageNet でプレトレインした ResNet-152 を使用して特徴 $\boldsymbol{v}_{ij}$ を得る． 次にサブショット $v_i$ の表現としてすべてのフレームを平均プーリングする．

```math
\boldsymbol{v}_i = \sum_j \boldsymbol{v}_{ij} \in \mathbb{R}^{7\times7\times2048}
```

各文 $s_i$ について，単語に分割し，プレトレインされたWord2Vecを適用後に position encoding（[PE](https://arxiv.org/abs/1503.08895)）を有する平均プールを行う．  

```math
s_i = \sum_j PE(s_{ij}) \in \mathbb{R}^{300}
```

最後に $v_i$ と$s_i$ に Compact Bilinear Pooling（[CBP](https://arxiv.org/abs/1606.01847)）を適用する．  

```math
E[i] = CBP(v_i,s_i) \in \mathbb{R}^{4096}
```

これをすべての $n$ 個のサブショットとテキストのペアに対して実行し，結果として2D映画埋め込み行列 $E \in \mathbb{R}^{n\times 4096}$ を得る．

## (b) Write Network
映画埋め込み行列 $E$ を入力とし，メモリテンソル $M\in\mathbb{R}^{m\times d\times 3}$ を出力として生成．($m=\lfloor ((n-1)/s_v^w+1)\rfloor$)  
人間が映画を理解するとき，イベントまたはエピソードの形でいくつかの隣接する発話およびシーンを関連づけていると考え，各メモリセルは隣接する映画埋め込みを関連付けるために CNN を利用する．  
フィルタサイズ $f_v^w$、フィルタチャネルの数 $f_c^w$、ストライド $s_v^w$ については実験で考察．

## (c) Read Network
質問 $q$ と $M$ から回答を生成する．  
Word2Vecベクトルを用いて質問文 $q$ のベクトル $\boldsymbol{q}$ を取得し，以下のように投影する．

```math
u=W_q\boldsymbol{q}+b_q
```

$W_q\in\mathbb{R}^{d\times3000}$ , $b_q\in \mathbb{R}^d$ はパラメータ．

次にメモリ $M$ とクエリ埋め込み $u$ を入力とし，次のようにして信頼スコアベクトル $o\in\mathbb{R}^d$ を生成する．  
まず、メモリ $M$ をクエリ依存 $M_q \in \mathbb{R}^{m\times d\times 3}$ に変換するために，$M$ の各メモリセルとクエリ埋め込み $u$ に CBP を適用する．

```math
M_q[i,;,j]=CBP(M[i,;,j],u)
```

ただし $i=1,...,m$ , $j=1,2,3$

CNNを利用して read ネットワークを実装する．映画に関する質問に正しく答えるためには一連のシーン全体をつなぎ合わせて関連付けることが重要であると考え，CNNアーキテクチャを使用してシーケンシャルメモリスロットのチャンクにアクセスする．  
フィルタサイズ $f_v^w$、フィルタチャネルの数 $f_c^w$、ストライド $s_v^w$ については実験で考察．  
出力としてメモリ $M_r\in \mathbb{R}^{c\times d\times 3}$ を得る．($c=\lfloor(m-1)/s_v^r + 1\rfloor$)


## (d) Question Answering
クエリ埋め込み $u$ とメモリ $M_r$ からアテンショ行列 $\boldsymbol{p}\in\mathbb{R}^{c\times3}$ を計算する．


```math
\boldsymbol{p}[i,j]=softmax(M_r[i,:,j]\cdot u)
```

$M_r$ の各メモリセルとアテンションベクトル $p$ から出力ベクトル $\boldsymbol{o}\in\mathbb{R}^d$ を計算する．

```math
$\boldsymbol{o}[i]=\sum_{j=1}^c\sum_{k=1}^3 M_r[j,i,k] p[j,k]$
```

次に、パラメータ $W_q$ と $b_q$ を共有して、式（4）の質問に対して行われた5つの応答候補文{a}の埋め込みを得る。 その結果、回答候補の埋め込み $g\in\mathbb{R}^{5\times d}$ を計算する。
$g$ と $o$ と $u$ の加重和の間の類似性を見つけることによって信頼ベクトル $\boldsymbol{z}\in\mathbb{R}^5$ を計算する。

```math
\boldsymbol{z}=softmax((\alpha o+(1-\alpha)u)^Tg)
```

```math
y=argmax_{i\in[1,5]}(\boldsymbol{z}_i)
```

## training
モデルの学習では予測値 $\boldsymbol{z}$ と真値の one-hot ベクトル $\boldsymbol{z}_{gt}$ との間のソフトマックスクロスエントロピーを最小化する．

# 評価実験
RWMN モデルを MovieQA ベンチマークのすべてのタスクで評価．

## Movie QAデータセット
![table1]  
各QAペアは5つの回答選択肢で構成されている．  
データセットには映画に関連する5種類のストーリーソースが用意されている．（videos, subtitles, DVS, scripts, plot synopses）  
MovieQAチャレンジはどの情報源を使用するかに応じて6つのサブタスクにわけられる．
(i) video+subtitle, (ii) subtitles only, (iii) DVS only, (iv) scripts only, (v) plot synopses only, (vi) open-ended.  
テキストのみのQAタスクでは視覚情報源　$\{v_1,...,v_n\}$ は与えられないので，CBPなしで $\{s_1,...,s_v\}$ を用いて式（1）の埋め込み $E$ を構成する．


## 定量的評価
![table2]  
RWMN-bag : 30ブートストラップされたデータセットでRWMNモデルを独立して学習し，平均予測を計算  
RWMN-ensemble : 異なるランダム初期化を持つ20個のモデルを個別に訓練し，平均予測を計算  


![table3]  
4つのテキストのみのタスクではストーリーに関する文章量 $n$ が異なる．Subtitle : $n=1,558$ , Script : $n=2,877$ ，DVS：$n=636$ , Plot Synopses : $n=35$．  
RWMN と MEMN2N を含むメモリネットワークのアプローチは Plot Synopses の性能が悪い．少ない文にの QA には，ストーリー理解の少ない単語/文のマッチング方法の方が性能が良い．  
DVS のみのタスクでは RWMN によるパフォーマンスの向上がより顕著なので，read/write ネットワークはハイレベルで抽象的なコンテンツを理解している．

## ablation result
![table4]  
$f_v$ が大きいほど，より多くのメモリがチャンクとして読み込まれる．$s_v$ が小さくなるか，$f_c,f_r$ が増えると、メモリブロックの総数が増加する．  
層の数が増加するにつれてメモリ相互作用の能力も増加するが，性能は悪化する．理由として MovieQA のデータセットのサイズが小さいから．  
$s_v$ が $f_c$ に比べて小さすぎたり大き過ぎたりするとパフォーマンスが低下する．

![fig3]  
Whyの質問の方が抽象化と高度な推論が必要．RWMNはWhyの質問のパフォーマンスが向上していることから，RWMNが高レベルの推論問題を扱う優位性を示唆している可能性がある．

## 定性的評価
![fig4]


# 議論はあるか
複雑なストーリー理解を必要とする他のQAタスクに適用してみる．   
ResNetやWord2Vec以外のビデオやテキストの表現方法を適用してみる．