Yikang Li 1, Nan Duan 2, Bolei Zhou 3, Xiao Chu 1, Wanli Ouyang 4, Xiaogang Wang 1  
1 : The Chinese University of Hong Kong, Hong Kong, China 2 : Microsoft Research Asia, China
3 : Massachusetts Institute of Technology, USA 4University of Sydney, Australia  
CVPR2018  
[HP](http://cvboy.com/publication/cvpr2018_iqan/) , [arXiv](https://arxiv.org/abs/1709.07192) , [pdf](http://cvboy.com/pdf/publications/cvpr2018_iqan.pdf) , [GitHub](https://github.com/yikang-li/iQAN)

# どんなもの？
Invertible Question Answering Network (iQAN) というVQAとVQGを統合したモデルを提案．  
提案されたデュアルトレーニングフレームワークは，多くのVQAアーキテクチャでモデルのパフォーマンスを向上させることができる．  
![fig1]

# 先行研究との差分


# abstract
提案された可逆双一次融合モジュールとパラメータ共有方式により、私たちのiQANはVQAとそのデュアルタスクVQGを同時に達成することができます。私たちの提案するデュアル・レギュラー（デュアル・トレーニングと呼ばれる）で2つのタスクを共同して訓練することにより、モデルはイメージ、質問と回答の相互作用をよりよく理解することができます。トレーニングの後、iQANは質問または回答のいずれかを入力として受け取り、回答を出力することができます。  
CLEVRとVQA2のデータセットで評価したところ、私たちのiQANは、ベースラインのMUTAN VQAメソッドのトップ1精度を1.33％と0.88％向上させることができました。また、提案されたデュアルトレーニングフレームワークは、多くの一般的なVQAアーキテクチャでモデルのパフォーマンスを一貫して向上させることができることも示しています。

# model
![fig2]  

## VQA
[MUTANモデル](https://arxiv.org/abs/1705.06676) をベースとする．

回答は以下のように計算される．  
$\hat{a} =(\Tau \times_1 \textbf{q}) \times_2 \textbf{v}_q$

ここで，$\Tau \in \mathbb{R}^{d_q\times d_v\times d_a}$ ，$\times_i$ は以下のように計算される．

$(\Tau \times_i \textbf{U})[d_1,...,d_{i-1},j,d_{i+1},...,d_N]=\sum_{d_i=1}^{D_i}\Tau [d_1...d_N]\textbf{U}[d_i,j]$

$\Tau$の複雑さを減らすために，以下のように記述する．  
$\Tau = ((\Tau_c \times_1 \textbf{W}_q)\times_2 \textbf{W}_v)\times_3 \textbf{W}_a$

式１は以下のように記述することができる．  
$\textbf{W}_q\in \mathbb{R}^{t_q\times d_q} , \textbf{W}_v\in \mathbb{R}^{t_v\times d_v} , \textbf{W}_a\in \mathbb{R}^{t_a\times d_a}$

$\Tau_c\in\mathbb{R}^{t_q\times t_v\times t_a}$

$\hat{\textbf{a}} = ((\Tau_c \times_1 (\textbf{W}_q\textbf{q}))\times_2 (\textbf{W}_v\textbf{v}_q))\times_3 \textbf{W}_a$

$\widetilde{\textbf{q}} = \textbf{W}_q\textbf{q}$ , $\widetilde{\textbf{v}}_q=\textbf{W}_v\textbf{v}_q$ とすると，  

$\widetilde{\textbf{a}} = (\Tau_c\times_1 \widetilde{\textbf{q}})\times_2 \widetilde{\textbf{v}}_q$

ここで $\widetilde{\textbf{a}}$ は $\hat{\textbf{a}}=\widetilde{\textbf{a}}^T\times\textbf{W}_a$ から answer feature とみなせる．


相互作用モデリングの複雑さと表現力のバランスをとるために  
$\Tau_c[:,:,k]=\sum_{r=1}^R \textbf{m}_r^k\otimes{\textbf{n}_r^k}^T$  
$\textbf{m}_r^k \in \R^{t_q} , \textbf{n}_r^k \in \R^{t_v}$

$\widetilde{\textbf{a}}[k]=\sum_{r=1}^R(\widetilde{\textbf{q}}^T\textbf{m}_r^k)(\widetilde{\textbf{v}}_q^T\textbf{n}_r^k)$

$\textbf{M}_r[:,k]=\textbf{m}_r^k , \textbf{N}_r[:,k]=\textbf{n}_r^k$ とすると

$\widetilde{\textbf{a}}=\sum_{r=1}^R(\widetilde{\textbf{q}}^T\textbf{M}_r)\odot(\widetilde{\textbf{v}}_q^T\textbf{N}_r)$


## VQG
トレーニング中に、生成された質問$q$が参照された$q^*$に類似するようなモデルを学習する．  

$\hat{w}_t=\text{argmax}\, p(w|v,w_0,...,w_{t-1})$

$w_i$は$i$番目の真値の単語．

$\widetilde{\textbf{q}}=\sum_{r=1}^R (\widetilde{\textbf{a}}^TM_r')\odot(\widetilde{\textbf{v}}_a^TN_r')$

$\widetilde{\textbf{a}}=\textbf{W}_a\textbf{a}$ , $\widetilde{\textbf{v}}=\textbf{W}_v\textbf{v}_a$

## Dual MUTAN
$\widetilde{\textbf{a}}^*=(\Tau_c\times_1\widetilde{\textbf{q}})\times_2\widetilde{\textbf{v}}$ 

$\widetilde{\textbf{q}}^*=(\Tau_c'\times_1\widetilde{\textbf{a}})\times_2\widetilde{\textbf{v}}'$

簡単にするために以下の性質を維持するようにする  
$\Tau_c'[:,i,:]=\Tau_c^T[:,i,:]$

$\widetilde{\textbf{a}}^*=(\Tau_c\times_1\widetilde{\textbf{q}})\times\widetilde{\textbf{v}}$


$\widetilde{\textbf{q}}^*=(\Tau_c\times_3\widetilde{\textbf{a}})\times\widetilde{\textbf{v}}$

以下の制約を追加する
```math
\left\{
\begin{array}{ll}
t_a = t_q = t \\
\Tau_c[:,i,:]=\Tau_c^T[:,i,:], i\in[i,t_v]
\end{array}
\right.
```


$\widetilde{\textbf{a}}^*=(\Tau_c\times_1\widetilde{\textbf{q}})\times\widetilde{\textbf{v}}$

$\widetilde{\textbf{q}}^*=(\Tau_c\times_1\widetilde{\textbf{a}})\times\widetilde{\textbf{v}}$


$\widetilde{\textbf{a}}^*=\sum_{r=1}^R (\widetilde{\textbf{q}}^TM_r)\odot(\widetilde{\textbf{v}}^TN_r)$

$\widetilde{\textbf{q}}^*=\sum_{r=1}^R (\widetilde{\textbf{a}}^TM_r)\odot(\widetilde{\textbf{v}}^TN_r)$

最終的な予測回答と質問は以下のように生成される  
$\hat{\textbf{a}}=\widetilde{\textbf{a}}^{*T}\times\textbf{W}_a$  
$\hat{\textbf{q}}=\widetilde{\textbf{q}}^{*T}\times\textbf{W}_q$

# Weight Sharing between Encoder and Decoder
Q/A のエンコーダーとデコーダーは逆変換とみなすことができるので，この特性を利用し，重み共有を行う． 

$E_a=W_a^T$

質問のエンコーダとデコーダは同じ単語の語彙を使用しているため，ＲＮＮの重みを共有する． 

# Duality Regularizer
予測された回答/質問表現は以下の形式を持つことが期待される：  
$\textbf{a}\approx \hat{\textbf{a}}= \phi(\textbf{q},\textbf{v}) \,\text{and}\, \textbf{q}\approx \hat{\textbf{q}} = \phi^*(\textbf{a},\textbf{v})$

Loss を計算するために以下の式を提案する．

```math
\text{smooth}_{L1}(x) = \left\{
\begin{array}{ll}
0.5*x^2 & if |x|<1 \\
|x|-0.5 & \text{otherwise}
\end{array}
\right.
```

$Loss = L_{(vqa)}(a,a^*) + L_{(vqg)}(q,q^*) + \text{smooth}_{L1}(\textbf{q}-\hat{\textbf{q}}) + \text{smooth}_{L1}(\textbf{a}-\hat{\textbf{a}})$

$L_{(vqa)}(a,a^*)$ : multinomial classification loss[3]
$L_{(vqg)}(q,q^*)$ : sequence generation loss[33]

# 評価実験
[VQA](https://visualqa.org/download.html)とCLEVR([HP](https://cs.stanford.edu/people/jcjohns/clevr/),[arXiv](https://arxiv.org/pdf/1612.06890.pdf),[GitHub](https://github.com/facebookresearch/clevr-dataset-gen))のデータセットで評価する．  
回答が'yes/no'や数字の場合は質問が作れないので，質問のタイプを制限する．  
VQAでは "what","where","who" で始まるタイプの質問のみを選択．  
CLEVRでは "what" で始まり，答えが数字ではない質問を選択．　　

![table1]  


![table2]  
![table4]  
Dual Training を他のVQAモデルに適用すると性能が上がる．  



![fig4]  
![fig5]  
VQAとVQGはそれぞれ異なるアテンションマップを生成するので，モデルの学習を助ける．


# まとめ
