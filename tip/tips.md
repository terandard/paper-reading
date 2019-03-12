Damien Teney, Peter Anderson, Xiaodong He, Anton van den Hengel
CVPR2018  
[arXiv](https://arxiv.org/abs/1708.02711) , [pdf](https://arxiv.org/pdf/1708.02711.pdf)  , [github](https://github.com/peteanderson80/bottom-up-attention)


# どんなもの？
2017年のVQAチャレンジで1位を獲得したモデルを説明．  
比較的単純なモデルではあるが，以下の特徴を用いて高いパフォーマンスを発揮した．
- 共通の単一ラベルsoftmaxの代わりに，質問ごとに複数の正解を可能にする **シグモイド出力(sigmoid output)** を使用
- 候補回答の分類ではなく，候補回答のスコアの回帰としてタスクを投げかける、**真のターゲットとしてのソフトスコア(soft scores as ground truth targets)** を使用
- すべての非線形層で **ゲート付き tanh 活性化関数(geted tanh activations)** を使用
- CNNのグリッド状の特徴マップではなく，領域固有の特徴を提供する **ボトムアップアテンションの画像特徴(image features from bottom-up attention)** を使用
- **プリトレインした候補回答の表現(pretrained representations off candidate answers)** を使用して出力層の重みを初期化
- **大規模なミニバッチ(large mini-batches)** を使用し，確率的勾配降下中にトレーニングデータを **smart shuffing** する．  



# 提案モデル
![model]

## question embedding
　質問をトークン化し，計算効率のために最大14語に制限．各単語をプリトレインされた[GloVe word embedding](https://nlp.stanford.edu/projects/glove/)を用いて300次元のベクトルに変換．GloVe に存在しない単語はゼロベクトルで初期化．14語未満の質問はゼロベクトルで補完．結果として得られるサイズは14×300であり，これを Recurrent Gated Unit（GRU）を用いて512次元に変換する．

## image features
　入力画像から Faster R-CNN で実装したボトムアップアテンションを用いてサイズ $K$×2048（$K$は画像位置の数）の特徴ベクトルを取得する．Faster R-CNN は [Visual Genomeデータセット](https://visualgenome.org/)のアノテーションを使用して画像の特定の要素に焦点を当てるように訓練．

## image attention
画像内の各場所 $i = 1...K$ について特徴ベクトル $v_i$ と質問 $q$を連結する．非線形層 $f_a$ (ゲート付き tanh )と線形層を通してアテンション重み $α_{i,t}$ を得る．アテンション重みはsoftmax関数を用いて正規化される．そして重み付けされた画像特徴$\widehat{v}$を得る．

```math
a_i = \omega_a f_a([v_i,q])
```

```math
\bf{\alpha} = softmax(a)
```

```math
\widehat{v} = \sum_{i=1}^K \alpha_i v_i
```

$\omega_a$ は学習パラメータ．

## multimodal fusion
質問（$q$）と画像（$v$）ベクトルは非線形層を通してアダマール積で組み合わせる．

```math
h = f_q(q) \circ f_v(\widehat {v})
```

## output classifier
　出力候補の回答は訓練セット内で8回以上出現する単語($N = 3129$)とする．VQAをマルチラベルの分類として扱う．マルチラベル分類器は $N$ 個の候補のそれぞれについてスコア $\widehat{s}$ を予測するために，非線形層 $f_o$ を通した $h$ を使用する．  

```math
\widehat{s} = \sigma(\omega_o f_o(h))
```

$\sigma$ はシグモイド関数，$\omega_o \in \mathbb{R}^{N\times 512}$ は学習パラメータ  
ソフトターゲットスコアを使用し，損失関数は以下の様に計算される．

```math
L = -\sum_i^M \sum_j^N s_{i,j} \log(\widehat{s}_{ij}) - (1-s_{ij})\log(1-\widehat{s}_{ij})
```

$i,j$ はそれぞれ $M$ 個のトレーニング質問と $N$ 個の候補回答．真値スコア $s$ は真値解答のソフト精度．上記の式は他のVQAモデルで一般的に使用されるソフトマックス分類器よりもはるかに効果的であることが分かった． 上記式の利点は2つある。第1に、シグモイド出力は、VQA v2データセットの場合と同様に、質問ごとに複数の正解を最適化することを可能にする．第2に、ターゲットとしてのソフトスコアの使用は、バイナリターゲットよりもわずかに豊富なトレーニング信号を提供します。なぜなら、それらは真値の注釈における時折の不確実性を捕捉するからです。  

## pretraining the classifier  
　2つの情報源からの候補回答に関する事前情報を使用して $\omega_o$ を初期化する．1つ目は GloVe を用いた回答単語の言語情報を使用する．対応するベクトルは行列 $\omega_o^{text}$ 内に置かれる．  
　2つ目は候補回答を表す画像から収集したビジュアル情報を使用する．Google画像を利用して候補回答ごとに10枚の画像を取得する．これらの画像を ImageNet で学習済みの ResNet-101 CNN に通し，平均プールによって 2048 サイズのベクトルを得る．このベクトルは行列 $\omega_o^{img}$ 内に置かれる。  
$\omega_o^{text}$ と $\omega_o^{img}$ を次の様に組み合わせる．

```math
\widehat{s} = \sigma(\omega_o^{text} f_o^{text}(h) + \omega_o^{img} f_o^{img}(h))
```

非線形変換 $f_o^{text}(h)$ および $f_o^{img}(h)$ は，$h$ をそれぞれ300および2048に変換する．


## training
確率的勾配降下を用いてネットワークを訓練する．[AdaDeltaアルゴリズム](https://arxiv.org/pdf/1212.5701.pdf)を使用．  
Visual Genome を追加のトレーニングデータとして使用．  
学習中に同じミニバッチでVQA v2のバランスの取れたペア（同じ質問に対する異なる画像と回答）を維持するためにトレーニングインスタンスのシャッフルを行う．

# 評価実験
## training data
トレーニングデータの考察．  
同じミニバッチでバランスの取れたペアを維持するようにトレーニングデータをシャッフルするとペアの精度は改善される．提案手法の目的がバランスの取れたペア間での違いの学習を改善することを目的とするからだと予想．  

![training_data]

## 4.2 question embedding
GloVeの次元数と1層のGRUを使用することによる結果の考察．  
ランダム初期化では、パフォーマンスが0：87％低下します。 事前訓練された埋め込みを伴うギャップは、モデルがより少ないトレーニングデータで訓練されるにつれてさらに大きくなる（図4.9セクション4.9）一方では、非VQAトレーニングデータを利用する利点を示している。 他方、十分に大きなVQAトレーニングセットがこの利点を完全に取り除く可能性があることを示唆している。 GRUを高度なオプション（後方、双方向、または2層GRU）に置き換えると、パフォーマンスが低下します。  
![question_embedding]

## 4.3 image feature
ResNetとの比較  
![image_feature]

## 4.4 image attention
画像のアテンションの考察  
ResNetよりもbottom-upアテンションを用いた方がよく，アテンション重みを複数セット使用しても良い結果が得られなかった．  
![image_attention]

## 4.5 output vocabulary
トレーニングセットに出現する候補回答の集合の考察  
トレーニング時に 8~12 回以上出現する語彙を用いると良いパフォーマンスを得る．  
![output_vocabulary]

## 4.6 output classifier
![output_classifier]

## 4.7 general architecture
非線形層の活性化関数と隠れ状態の次元数の考察  
活性化関数はゲート付き tanh が良く，次元数は大きいほど良いが分散が大きいので保証が出来ない．  
![general_architecture]


## 4.8 mini-batch size
ミニバッチサイズの考察  
ミニバッチサイズは小さいよりも大きい方が良い  
![mini-batch_size]

## 4.9 training set size
パフォーマンスとトレーニングデータの量との関係  
わずか10％のデータで妥当なパフォーマンスを得ており，より多くの珍しい概念をカバーするためには指数関数的に大きなデータセットが必要．  
単語の埋め込みと分類器をプリトレインするためのエキストラデータの使用は常に有益．  

## 4.10 ensembling
アンサンブルの結果  
アンサンブルサイズ 30 で最良の結果を獲得．  
2〜5の小さなアンサンブルであってもパフォーマンスが大幅に向上する．  


## 他手法との比較


# 議論はあるか
パフォーマンスはアーキテクチャとハイパーパラメータの選択に非常に依存している．  
実装が容易なもの(ex:large mini-batch size，シグモイド出力)で性能が上がるものもある．  
現在のモデルでは言語構造を理解し効果的に利用できていない．  
より大きなデータセットを収集するよりも，他の情報源を組み込んだ非VQAデータセットを活用することが有望な方向であると考えている．    
