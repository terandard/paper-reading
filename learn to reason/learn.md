Hu R, Andreas J, Rohrbach M, Darrell T, Saenko K  
CVPR2017  
[arXiv](https://arxiv.org/abs/1704.05526) , [pdf](https://arxiv.org/pdf/1704.05526.pdf) , [GitHub](https://github.com/ronghanghu/n2nmn) ,[HP](http://ronghanghu.com/n2nmn/)

# どんなもの？
質問文は構造的なものであり，モジュール単位の副問題に分解することで，簡単に応えることができる．  
既存手法の Neural Module Networks ([NMN](https://arxiv.org/pdf/1511.02799.pdf)) は，質問を部分構造に構文解析し，それぞれが１つのサブタスクを解く小さなモジュールから質問特有のネットワークを組み立てた．  
NMN を拡張した End-to-End Module Networks (N2NMNs) を提案．  
![fig1]

# 既存研究との差分
NMN は脆弱な既製のパーサに依存しており，データから学ぶよりもこれらのパーサによって提案されたモジュール構成に制限されている．  
提案手法では，パーサを使わずにインスタンス固有のネットワークレイアウトを直接予測する．


# model
![fig2]  

## Attentional neural modules
ニューラルモジュール $m$ は， $y=f_m(a_1,a_2,...;x_{vis},x_{txt},\theta_m)$ によってパラメータ化される．   
$a_1,a_2,...$ : 入力(アテンションマップ)  
$\theta_m$ : 内部パラメータ  
$x_{vis},x_{txt}$ : 入力から計算された画像と質問特徴  
$y$ : 出力(画像アテンションマップまたは候補回答の確率分布)

モジュールの詳細  
![table1]  

各モジュール $ｍ$ について，$T$ 個の質問語にわたるアテンションマップ $\alpha_i^{(m)}$ を予測し，各モジュールに対するテキスト特徴を得る：

$x_{txt}^{(m)}=\sum_{i=1}^T\alpha_i^{(m)}w_i$

実行時には，モジュールはレイアウト $l$ に従って組み立てられる( $f_{m2}(f_{m4}(f_{m1}),f_{m3}(f_{m1},f_{m1}))$ )



## Layout policy with sequence-to-sequence RNN
各質問に合わせた最適な推論構造を予測する．  
入力質問 $q$ に対して，layout policy は確率分布 $p(l|q)$ を出力し，レイアウト $l$ を得る．
次に予測レイアウト $l$ に従ってニューラルネットワークが組み立てられ，質問に対する回答が出力される．

考えられるすべてのレイアウト $l$ は構文木として表すことができるので，[Reverse Polish Notation](https://www.ams.org/journals/mcom/1954-08-046/S0025-5718-1954-0061484-4/S0025-5718-1954-0061484-4.pdf) を使用して $l=\{m^{(t)}\}$ の様に表す．  

![fig3]

その後，attentional RNN を使用  
質問内のすべての単語に対して多層LSTMを用いて $[h_1,h_2,...,h_T]$ にエンコードする． 

デコーダは，エンコーダと同じ構造を持つがパラメータが異なるLSTMネットワークを使用する．
デコーダの時間ステップ $t$ において，位置 $i$ における入力単語のアテンション重みは次のように予測される：

$u_{ti}=v^T\tanh(W_1h_i+W_2h_t)$  
$\alpha_{ti}=\frac{\exp(u_{ti})}{\sum_{j=1}^T\exp(U_{tj})}$

$h_i$ : エンコーダーの時間ステップ$i$における出力  
$h_t$ : デコーダーの時間ステップ$t$における出力

コンテキストベクトル : $c_t = \sum_{i=1}^T\alpha_{ti}h_i$  
$p(m^{(t)}|m^{(1)},m^{(2)},...,m^{(t-1)},q) = \text{softmax}(W_3h_t+W_4c_t)$

$p(m^{(t)}|m^{(1)},m^{(2)},...,m^{(t-1)},q)$ からサンプリングして次のトークン $m^{(t)}$ を離散的に取得し，そのテキスト入力 $x_{txt}^{(m)}$ を構築

$p(l|q) = \prod_{m^{(t)}\in l} p(m^{(t)}|m^{(1)},m^{(2)},...,m^{(t-1)},q)$


## End-to-end training
ニューラルモジュールとレイアウトポリシーを共同で最適化するための強化学習アプローチを提示する  

$\theta$ : モデルの全パラメータ  
$L(\theta)=E_{l~p(l|q;\theta)}[\tilde{L}(\theta,l;q,I)]$

$\tilde{L}(\theta,l;q,I)$ : 出力回答スコアに対するsoftmax loss

モンテカルロサンプリングを使用して  
$\Delta_\theta L= E_{l~p(l|q;\theta)}[\tilde{L}(\theta,l)\Delta_\theta\log p(l|q;\theta)+\Delta_\theta\tilde{L}(\theta,l)]$  

$\Delta_\theta L\approx 1/M\sum_{m=1}^M(\tilde{L}(\theta,l_m)\Delta_\theta\log p(l_m|q;\theta)+\Delta_\theta\tilde{L}(\theta,l_m))$  

実験では $M=1$ を使用

### Behavioral cloning from expert polices

学習をより簡単にするために，質問からレイアウトを予測する expert policy $p_e(l|q)$ を導入する．    
まず $p_e$ からの模倣によってモデルをプレトレインする．これは，$p_e$ と $p$ との間の KL-divergence $D_{KL}(p_e||p)$ を最小化し，同時に $p_e$ から得られる $l$ を用いて $\tilde{L}(\theta,l; q, I)$ を最小化することによって行うことができる．$p_e$ を模倣することによって初期値を学習した後 $\Delta_\theta L$ でさらに訓練される．


# 評価実験

## SHAPE
244個の固有の質問を含む15616個の画像と質問のペアで構成． ([NMN](https://arxiv.org/abs/1511.02799)が作成)  

![table2]  
![fig4]  

## CLEVER

![table3]  
![fig5]  

既存手法では質問の中に長い推論チェーンがあると，中間推論結果を記憶するための短期記憶が不足するので性能があまり良くない．  
提案モデルは各質問に合わせた推論レイアウトを動的に予測するのでこれらの問題に対応出来る．また中間推論結果は各モジュールの出力として保存され，他のモジュールによってさらに使用できる．

## VQA
![table4]  

![fig7]  

根底にある推論手順を明示的に見ることができるので，提案モデルはより解釈しやすくなっている．  
