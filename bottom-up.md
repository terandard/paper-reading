Peter Anderson，Xiaodong He，Chris Buehler，Damien Teney，Mark Johnson，Stephen Gould，Lei Zhang
CVPR2018
[pdf](https://arxiv.org/pdf/1707.07998.pdf) , [arXiv](https://arxiv.org/abs/1707.07998) , [git](https://github.com/peteanderson80/bottom-up-attention)

## どんなもの？
ボトムアップとトップダウンを組み合わせたアテンションメカニズムを提案．  
MSCOCOテストサーバーでCIDEr / SPICE / BLEU-4スコアはそれぞれ117.9，21.5，36.9を獲得．  
2017年のVQAチャレンジで第1位を獲得．

## 先行研究との差分
典型的なアテンションモデルでは，等サイズの画像領域の均一グリッド(下図左)に対応する CNN 特徴上で動作するのに対して，提案手法ではオブジェクトおよび他の顕著な画像領域（下図右）のレベルでアテンションを計算することができる．  
![Screen Shot 2018-11-09 at 11.53.26.png](https://qiita-image-store.s3.amazonaws.com/0/312111/05ad117f-22dc-5c2b-0dbe-462264985a92.png)


## モデルの説明

$`k`$個の画像特徴を $`V =\\{v_1,...,v_k\\}, v_i \in \mathbb{R}^D`$とする．

### bottom-up attention model
Faster R-CNNを使用．プリトレインするために，ImageNetで学習済みのResNet-101を用いてFaster R-CNNを初期化する．その後，[Visual Genome データセット](https://visualgenome.org/)で訓練する．トレーニングに98K，validationとテストにそれぞれ5Kの画像を使う．  
下図は Faster R-CNN ボトムアップアテンションモデルの出力例．  
![Screen Shot 2018-11-09 at 11.54.40.png](https://qiita-image-store.s3.amazonaws.com/0/312111/de355a8c-84dc-af72-cf46-07b0945b6cf4.png)

### caption model
2層の LSTM を用いる．1層でトップダウンアテンションを学習(attention LSTM)，2層で言語モデルを学習(language LSTM)．  
モデルの図  
![Screen Shot 2018-11-12 at 17.37.00.png](https://qiita-image-store.s3.amazonaws.com/0/312111/3a23c2fd-58ae-368c-bea0-45edf8ef2d84.png)

#### attention LSTM
各時間ステップでの attention LSTM の入力ベクトルは，平均プール画像特徴 $`\bar{v}=\frac{1}{k}\sum_iv_i`$，前の時間ステップの language LSTM の出力$`h_{t-1}^2`$，前の時間ステップで生成された単語からなる．

```math
x_t^1 = [h_{t-1}^2 , \overline{v} , W_e\Pi_t]
```

$`W_e \in \mathbb{R}^{E \times | \Sigma |}`$ は語彙集合$`\Sigma`$の word embedding matrix  
$`\Pi_t`$ は時間ステップ$`t`$の入力単語の one-hot ベクトル  

attention LSTMの出力$`h_t^1`$が得られると，各時間ステップ$`t`$において，$`k`$個の画像特徴$`v_i`$の各々に対して正規化された attention 重み$`\alpha _{i,t}`$を生成する．

```math
a_{i,t} = \omega_a^T \tanh(W_{va}v_i+W_{ha}h_t^1)
```

```math
\alpha_t = softmax(a_t) \tag{1}
```

$`W_{va} \in \mathbb{R}^{H \times V} , W_{ha} \in \mathbb{R}^{H \times M} , \omega_a \in \mathbb{R}^H`$は学習したパラメータ  

言語LSTMの入力に使う attention 重み付き画像特徴は次のように計算される．  

```math
\widehat{v}_t = \sum_{i=1}^K \alpha_{i,t}v_i \tag{2}
```

#### language LSTM
language LSTM への入力は，attention LSTM の出力と attention 重み付き画像特徴からなる．

```math
x_t^2 = [\widehat{v}_t , h_t^1]
```

文の単語($y_1,...,y_t$)を参照する表記を$y_{1:T}$をすると，各時間ステップ$t$において，可能な出力単語に対する条件付き分布は次のように計算される．

```math
p(y_t|y_{1:t-1}) = softmax(W_p h_t^2 + b_p)
```

$W_p \in \mathbb{R}^{|\Sigma|\times M} , b_p \in \mathbb{R}^{|\Sigma|}$は学習した重みとバイアス

最終的な出力文に対する分布は条件付き分布の積として計算される．

```math
p(y_{1:T}) = \prod_{t=1}^T p(y_t|y_{1:t-1})
```

真の文$y_{1:T}^\ast$とキャプションモデルのパラメータ$\theta$を用いて交差エントロピーロスを計算．

```math
L_{XE}(\theta) = - \sum_{t=1}^T \log(p_\theta ( y_t^\ast | y_{1:t-1}^\ast))
```

最近の研究との比較のために，CIDErに最適化された結果も提示する．クロスエントロピーで訓練されたモデルから初期化することで，次のスコアを最小限に抑える．

```math
L_R(\theta) = - E_{y_{1:T}~p_\theta}[r(y_{1:T})]
```

```math
\nabla_\theta L_R(\theta) \approx -(r(y_{1:T}^s) - r(\widehat {y}_{1*T})) \nabla_\theta \log p_\theta (y_{1:T}^s)
```

$r$はCIDErのスコア関数．
### vqa model
![Screen Shot 2018-11-09 at 11.55.10.png](https://qiita-image-store.s3.amazonaws.com/0/312111/637a57c8-c3f7-a3f2-2116-c3b1dbebb4ea.png)

キャプションモデルと同様にソフトトップダウンアテンションメカニズムを使用する．
提案手法では，各質問は学習された単語埋め込みを用いて表現されたGRUの隠れ状態$q$として最初に符号化される。 方程式3と同様に、GRUの出力$q$が与えられると、以下のように、$k$個の画像特徴$v_i$のそれぞれについて非標準化されたアテンション重み$a_i$を生成する。

```math
a_i = \omega_a^Tf_a([v_i,q])
```

キャプションモデルと同様に式(1),(2)を用いて画像特徴$\widehat{v}$を計算する．
可能な出力応答$y$に対する分布は次のように計算される．

```math
h=f_q(q) \circ f_v(\widehat{v})
```

```math
p(y) = \sigma(W_of_o(h))
```

VQAモデルの詳細については [Tips and Tricks for Visual Question Answering](https://arxiv.org/pdf/1708.02711.pdf) を参照



## 評価実験

　ボトムアップのアテンションの影響を評価するために，キャプショニングとVQAの両方の実験で，ベースライン（ResNet）と完全モデル（Up-Down）を評価する．ベースラインは、ボトムアップアテンションメカニズムの代わりに ImageNet で学習済みのResNet CNNを使用．
　画像キャプショニング実験では，Resnet-101の出力をバイリニア補間を使用して10×10の固定サイズにリサイズする．VQA実験では，サイズを変更した入力画像をResNet-200でエンコードする．別の実験では，元のサイズ14×14から7×7，1×1の空間出力のサイズを変化させる効果を評価する．

### キャプションの実験結果

[MSCOCO 2014キャプションデータセット](http://cocodataset.org/#home)を使用．
キャプションの品質をSPICE，CIDEr，METEOR，ROUGE-L，BLEUで評価．

MSCOCO Karpathy split テスト結果．
![Screen Shot 2018-11-10 at 13.20.35.png](https://qiita-image-store.s3.amazonaws.com/0/312111/35d4dc85-09e9-4bdc-b9bc-e56af79c777e.png)

MSCOCO Karpathy split テストに関する SPICE Fスコアの内訳．
![Screen Shot 2018-11-10 at 13.21.01.png](https://qiita-image-store.s3.amazonaws.com/0/312111/7277ad6b-05f7-1202-dec4-786a84419c06.png)

オンライン MSCOC テストサーバでのキャプション結果．提出時（2017年7月18日）にはテストサーバーに提出されていた他の手法よりも優れていた．
![Screen Shot 2018-11-10 at 13.37.45.png](https://qiita-image-store.s3.amazonaws.com/0/312111/569977ae-d0c4-6c65-d20f-f8a69d334c99.png)
キャプション例
![Screen Shot 2018-11-15 at 13.00.32.png](https://qiita-image-store.s3.amazonaws.com/0/312111/26175210-7270-998a-2156-21a31c2b4382.png)


### VQAの実験結果

[VQA v2.0データセット](http://visualqa.org/download.html)を使用．質問は最大14語に制限．回答候補の集合は訓練集合において8回以上出現する正解に限定し，その結果出力語彙サイズは3,129だった．

VQA v2.0の validation セットでの結果．
![Screen Shot 2018-11-10 at 13.38.10.png](https://qiita-image-store.s3.amazonaws.com/0/312111/a25d0782-34bd-6481-d387-6fba7b88bef1.png)

VQA 2.0テストサーバでの結果．2017年のVQAチャレンジにおいて第1位を達成.
![Screen Shot 2018-11-10 at 13.38.19.png](https://qiita-image-store.s3.amazonaws.com/0/312111/b772d18a-fecc-1f8c-b73a-81509239557a.png)
VQAの出力例
![Screen Shot 2018-11-15 at 13.00.12.png](https://qiita-image-store.s3.amazonaws.com/0/312111/282d5970-ffe5-a458-8dc7-5cbd3b33f48f.png)


## 定性的評価
### attentionの差
上がベースライン，下が提案手法．ベースラインではバスルームからトイレを連想して間違えているが，提案手法ではソファーを認識して正しいキャプションを生成している．
![figure:7png](https://qiita-image-store.s3.amazonaws.com/0/312111/000973d1-adbd-caa4-543c-327dccb36e86.png)

### キャプション

![figure:8](https://qiita-image-store.s3.amazonaws.com/0/312111/9409fc97-06e0-c7fc-4268-32dd3c602f64.png)

![figure:9](https://qiita-image-store.s3.amazonaws.com/0/312111/c858480c-29f1-faaf-113f-c139f44aad22.png)

### VQA
成功例
![figure:10](https://qiita-image-store.s3.amazonaws.com/0/312111/8e863893-cfbc-776a-7862-39343e065d05.png)

失敗例．
![figure:11](https://qiita-image-store.s3.amazonaws.com/0/312111/8d8f1666-ea88-1692-b3d9-d8240c478dfd.png)
