Qingxing Cao, Xiaodan Liang, Bailing Li, Guanbin Li, Liang Lin  
CVPR2018  
[arXiv](https://arxiv.org/abs/1804.00105),[pdf](https://arxiv.org/pdf/1804.00105.pdf)

# どんなもの？
adversarial attentionモジュールと residual compositionモジュールからなる Adversarial Composition Modular Network（ACMN）と呼ばれる新しい推論ネットワークを提案．  
 

# 先行研究との差分
従来のモデルは有効なレイアウトを得るために注釈または hand-crafted rule に依存している．
ACMNモデルは質問から生成される一般的な依存関係解析木に対して解釈可能な推論プロセスを自動的に実行することができる  
特定のレイアウトを取得するために複雑な手作りのルールや根拠のある注釈を必要としない，一般的で解釈可能な推論VQAフレームワークを目指す


# model
[fig2]  
$Q$ と $I$ が与えられると、ACMNモデルは答え $y$ とそれに対応する説明可能なアテンションマップを予測することを学ぶ．
最初に与えられた$Q$の構造レイアウトを既存の [off-the-self universal Stanford Parser](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf) を使用してツリー構造に解析することによって生成する．  
計算量を減らすために，名詞ではない葉ノードを取り除き，“nominal modifier”(ex:(left, object)) , “nominal subject”(ex:(is,color)) などの依存関係のラベルを modifier relation "M"と clausal predicate relation "P"の2つのクラスに分類する．

## Modifier Relation and Clausal Predicate Relation
最も広く使用されている head-dependent 関係セットの1つは Universal Dependencies([UD](https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf)) で，9つのカテゴリーに分類できる合計42の関係を持つ．  
この中でよく使われる関係はそのうちの2つだけに集中している  
![fig3]

![table1]  


## Adversarial Attention Module
[fig2b]  
関係が $M$ である子ノードをフィルタリングし，親ノード $x$ に対して Adversarial Attention Module を実行する．  
すべての視覚的証拠を効果的にマイニングするために，各ステップで各親ノードが子ノードのアテンション領域をマスクすることによって新しい領域を探索するように強制する．    
各ノード $x$ の入力 $att_{in}$ は子ノード$x_i$ の $\{att_i^c\}$ の合計から得られる．adversarial mask は $att_{in}$から1を引いた後にReLU層を通して生成．
次に mask を使用して $v$ を重み付けする．
最後に，入力単語 $w$ と重み付けされた $v$ に基づいて $att_{out}$を出力し，Softmaxを適用して正規化する．  
ノード $x$ の視覚的表現$h'$は，$att_{out}$が与えられたときの$v$内の各グリッド特徴の重み付き合計によって生成される．  
関係が $M$ であるノードはより具体的なオブジェクトを参照することによって親ノードを修正できるため，より正確なアテンションマップを生成する．

## Residual Composition Module
[fig2c]  
述語関係Pを持つ子ノードの隠れ特徴 $\{h_i^c\}$ の合計を $h_{in}$ とし，次に抽出された $h'$ と $h_{in}$ を連結し，最後に単語埋め込み $w$ と組み合わせて $h_{out}$ を生成する．  
述語関係Pは，親ノードが子ノードの述語であることを示しているので，述語が与えられた表現を強調するために子ノードの特徴を統合する．


## モデルの定式化

```math
att_{in}=\sum_{(x,x_i^c)\in M} att_i^c
```

```math
h_{in}=\sum_{(x,x_i^c)\in P} h_i^c
```

```math
att_{out}=f_a(att_{in},v,w)
```


```math
h'=att_{out}*v
```

```math
h_{out}=f_h([h_{in},h'],w)+\sum_i h_i^c
```



予測回答 $y$ はルートノードの出力特徴 $h_{root}$ を3層のMLPを通して得られる．



# 評価実験
## CLEVR
![table2]  
ACMNはデータセット固有のレイアウトを使用せずに既存モデルを大幅に上回っており，優れた汎化能力がある．

![fig4]  

## Sort-of-CLEVR
![table3]  
Ours-w/o residual : residual composition モジュールを取り除いたモデル  
Ours-DualPath : residual composition モジュールの入力について，前の隠れ表現hをすべて連結し，それらを256次元の特徴ベクトルに射影するために追加のFC層を使用  
一般的な依存関係解析ツリーのノードには重複した情報が含まれる可能性があるため，余分なFC層が性能を低下させる．
Ours-relocate : adversarial attention モジュールを [Relocate モジュール](https://arxiv.org/pdf/1704.05526.pdf)に変更
Ours-concat : 画像特徴とアテンションマップを concat 

![fig5]

## VQAv2
![table4]
提案モデルはデータ拡張や事前学習済み分類器などのトリックを適用していないので，1stより少し低い．

