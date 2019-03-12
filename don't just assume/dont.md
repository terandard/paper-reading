Agrawal A, Batra D, Parikh D, Kembhavi A  
CVPR2018  
[arXiv](https://arxiv.org/abs/1712.00377), [pdf](https://arxiv.org/pdf/1712.00377.pdf) , [GitHub](https://github.com/AishwaryaAgrawal/GVQA)

# どんなもの？
今日のVQAモデルは画像の根拠が欠けていることが多くの研究によってわかっている．  
そこで，候補回答の分布がトレーニングとテストで異なる Visual Question Answering under Changing Prior(VQA-CP v1 ,VQA-CP v2) という新しいデータの分割方法を提案．  
また，トレーニングの回答分布に依存しない Grounded Visual Question Answering (GVQA)を提案．  
![fig1]

## 先行研究との差分
以前のVQAアプローチでは $(I,Q)$ を $(A)$ に直接マッピングしていた

GVQAはVQAのタスクを2つのステップに分けます。  
GVQAは、VQAの質問が2つの重要な情報を提供するという直感によって動機づけられています。
（1）何を認識すべきか？あるいは、質問に答えるために画像のどの視覚的概念を推論する必要があるか（例えば、「プレートの色は何ですか？」は画像のプレートを見ることを必要とします）、
（2）何を答えるべきか？ あるいは、もっともらしい答えのスペースはどれか（たとえば、「色は…？」という質問には色の名前を付けて答える必要があります）。  

解釈可能な中間出力を生成するという点で、既存のVQAモデルよりも透過的です。

# abstract
次に、主にトレーニングデータのプライアに頼ることによってモデルが「不正行為」をしないように設計された、アーキテクチャ内の帰納的バイアスと制限を含む、新しい Grounded Visual Question Answering (GVQA)を提案します。具体的には、GVQAは、与えられた質問に対するもっともらしい回答空間の識別から、画像内に存在する視覚的概念の認識を明確に解きほぐし、モデルが異なる回答の分布にわたってよりロバストに一般化できるようにします。 GVQAは既存のVQAモデル -  Stacked Attention Networks（SAN）をもとに構築されています。GVQAは、オリジナルのVQA v1およびVQA v2データセットでトレーニングおよび評価された場合、SANを補完する強みを提供します。  
最後に、GVQAは既存のVQAモデルよりも透明で解釈しやすいものです。

# introduction
我々はまた、主に訓練データ内の前任者に頼ることによって「不正行為」を防止するように設計された、アーキテクチャにおける帰納的バイアスおよび制限を含む、新しい視覚的根拠質問応答（GVQA）モデルを提案する（セクション5）。 

私たちの仮説は、これら2つの役割を明確に区別しないモデル（これは文献のほとんどの既存モデルの場合です）は、これら2つのシグナルを混同する傾向があるということです。 彼らは質問と回答のペアから、プレートのもっともらしい色が白であることを学び、テスト時には、質問が関係している画像内の特定のプレートよりもこの相関に頼ります。 GVQAは回答空間の予測から視覚的概念の認識を明確に解きほぐします。

GVQAは既存のVQAモデル -  Stacked Attention Networks（SAN）[37]から構築されています。 私たちの実験は、GVQAが私たちの提案したVQA-CPデータセット上のすべてのタイプの質問においてSANよりかなり優れていることを示しています。 興味深いことに、いくつかのケースでは、Multimodal Compact Bilinear Pooling（MCB）[9]などのより強力なVQAモデルよりも優れています。 また、GVQAが元のVQAデータセットでトレーニングおよび評価されると、SANを補完する強みを発揮することも示します。 最後に、GVQAは、既存のVQAモデルとは異なり、解釈可能な中間出力を生成するという点で、既存のVQAモデルよりも透過的です。

# VQA-CP : Dataset Creation and Analysis
VQA-CP の分割は，質問タイプごとの回答分布がトレーニングとテストで異なるように作成する．
画像の分布は変更しない．


## Question Grouping
同じ質問タイプと正解を持つ質問をまとめる．  
- ex : {'What color is he dog?', 'white'} と {'what color is the plate?', 'white'} はグループ化されるが，{'What color is he dog?', 'black'} は別のグループ． 

このグループ化は，VQAの train と val のQAペアをマージした後に行われる．

## Greedily Re-splitting
train で test の concepts を最大限に網羅するようにしながら，同じ質問タイプと正解を持つ質問が test と train の間で繰り返されないようにする．  

concepts : 質問タイプとそのグループに属する正解におけるすべてのユニークな単語のセット

上で作成したすべてのグループをループ処理し，そのグループがすでに train に割り当てられていない限り，現在のグループを test に追加する．  
次にまだ割り当てられていないグループから，セット内の concepts の大部分をカバーするグループを選択し，そのグループを train に追加する．  
test がデータセットの 1/3 になったら残りのグループを train に追加する．

VQA-CP v2 train には ~121Kの画像，~483Kの質問，~4.4Mの回答が含まれている．test には ~98Kの画像，~220Kの質問，~2.2Mの回答が含まれている．

![fig6]



## 既存モデルの結果
- per Q-type prior [5]: Predicting the most popular training answer for the corresponding question type.
- Deeper LSTM Question (d-LSTM Q) [5]: Predicting the answer using question alone (“blind” model).
- Deeper LSTM Question + normalized Image (d-LSTM Q + norm I) [5]: The baseline VQA model.
- Neural Module Networks (NMN) [3]: The model designed to be compositional in nature.
- Stacked Attention Networks (SAN) [37]: One of the widely used models for VQA.
- Multimodal Compact Bilinear Pooling (MCB) [9]: The winner of the VQA Challenge (on real image) 2016.

![table1]


# GVQA model
GVQAはVQAのタスクを2つのステップに分ける  
- Look : 質問に答えるのに必要な object/image パッチを見つけ，パッチの視覚的概念を認識する．  
- Answer : 質問から適切な回答のスペースを特定し，どの概念が適切かを考慮して，認識された視覚的概念のセットから適切な視覚的概念を選択する． 

ex : ‘What color is the dog?’ -> 答えは色の名前であるべき   
犬に対応する画像内のパッチを見つける．
次に様々な視覚的概念('dog','black'など)を認識し，色に対応する概念である'black'を出力

GVQAにおける別の新規性は、yes/no の質問に答えることを視覚的検証タスクとして扱うということ．すなわち，質問に記載された概念の視覚的な 存在/不在 を検証すること．  

ex: 'Is the person wearing shorts?’  
視覚的存在を検証する必要がある概念が 'shorts' であることを識別し，画像内で shorts を認識するか否かに応じて 'yes','no' と答える

![fig3]

質問は最初に Question classifier によって'yes/no'または 'non yes/no' に分類される．   

'non yes/no'の場合
- CNNから抽出された画像特徴と質問抽出器によって与えられた $Q_{main}$ を入力として受け取る Visual Concept Classifier (VCC)
- 質問全体を入力とする Answer Cluster Predictor（ACP）
- VCC および ACP の出力は回答を生成する Answer Predictor(AP) に供給される．

'yes/no' の場合
- VCC（'non yes/no'と同様）
- 質問全体を入力とする Concept Extractor（CE） 
- VCCとCEの出力は yes/no を予測する Visual Verifier (VV) に供給される． 

## VCC
質問に答えるために必要なイメージパッチを見つけ，見つけたパッチに関連する一連の視覚的概念を作成する．  
これはStacked Attention Networks（[SAN]）に基づく2ホップのアテンションモジュールとそれに続くバイナリ概念分類子のスタックから構成される．  
![san]


質問タイプごとの answer priors の暗記を防ぐために，質問は最初に language Extractor を通過して質問タイプの部分文字列（ex：'What of'）を削除した $Q_{main}$ を生成し，LSTMを使用して埋め込んでからアテンションモジュールに送る．  
マルチホップアテンションは、その領域に対するアテンションの程度に対応する重みを用いて、VGG-Netからの画像領域の特徴の重み付き線形結合を生成する。 これに続いて、一連の全結合(FC)層と、約2000個の binary concept classifiers のスタックとが続く。 

VCCはすべての概念について binary logistic loss で訓練される．

一連のVCC概念は，QAペアをトレーニングし，最も頻繁なペアを保持しながら，回答に関連するオブジェクトと属性を抽出することによって構築される．
次に，オブジェクトの概念は単一のグループにグループ化され，属性の概念はGlove埋め込み空間でのK-meansクラスタリングを使用して$C$クラスタに分類される．
概念分類器を訓練するのに必要とされる負のサンプルを生成する目的のために、概念クラスタ化が必要とされる。（概念分類子の場合、肯定的なサンプルは、質問または回答のいずれかにその概念を含むものです）
質問は画像に存在しないオブジェクトや属性を示すものではないので、以下の仮定を使用して否定的なデータが生成されます。
- 質問に答えるために必要とされる注目画像パッチは、その中に少なくとも１つの主要なオブジェクトを有する。
- 各オブジェクトは、各属性カテゴリから最大で１つの支配的な属性を有する（例えば、バスの色が赤であれば、他のすべての色に対する否定的な例として使用することができる）

これらの仮定を前提として、クラスター内の概念がポジティブとして扱われる場合、そのクラスター内の他のすべての概念はネガティブとして扱われます。 トレーニング中に各質問に対してすべての概念クラスタのサブセットのみがアクティブ化され、これらのアクティブ化されたクラスタのみが損失の原因となることに注意してください。

## Question Classifier
入力された質問 Q をGlove埋め込み層，LSTM層，FC層を使用して yes/no, non yes/no の2つのカテゴリに分類する．
yes/no の質問はCEに入力され，non yes/no の質問はACPに入力される

## Answer Cluster Predictor (ACP) 
予想される回答の種類（オブジェクト名，色，番号など）を識別する．Glove埋め込み層とLSTM，質問を $C$ クラスタの1つに分類するFC層で構成される．
クラスタは k-means クラスタリングで作成．Glove で embedding した回答を 1000 クラスに分類．

## Concept Extractor (CE) 
POSタグベースの抽出システムを使用して，画像内で視覚的な存在を確認する必要がある質問の概念を抽出する．ex : ‘Is the cone green?’ の場合 ‘green’を抽出  
抽出された概念はGloveによって埋め込まれ，FC層がVCCの概念と同じ空間に変換することで，それらを VV によって組み合わせることができるようにする．

## Answer Predictor (AP)
答えを予測する．ACPのカテゴリはVCCの概念クラスタに対応している．この配列が得られると，ACPの次元をそれぞれのVCCクラスタの次元に関係する位置に単にコピーすることによって，ACPの出力はVCC出力と同じ次元に簡単にマッピングできる．  
得られたACPはVCCに要素ごとに追加され，続いてFC層とsoftmaxによって998 VQA回答カテゴリ(トップ1000のトレーニング回答 - 'yes','no')の分布が得られる．

## Visual Verifier (VV)
VCCの予測における概念の有無を検証する．  
CEはVCCに要素ごとに追加され，続いてFC層とsoftmaxによって'yes','no'のカテゴリにわたる分布が得られる．

# 評価実験

## Experiments on VQA-CP v1 and VQA-CP v2
![table1]  
![table2]  


## Role of GVQA Components
![table3]
CE が重要

## Experiments on VQA v1 and VQA v2
![table6]


