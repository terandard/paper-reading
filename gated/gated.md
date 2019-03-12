Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, Ming Zhou , ACL2017  
[pdf](http://www.aclweb.org/anthology/P17-1018)

# どんなもの？
Stanford Question Answering Dataset([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/))におけるSingleモデル，Ensembleモデルの両方で1位を獲得したモデル  
SQuADの例：  
![table1]

# 先行研究との差分
従来のAttentionベースのRNNでは，QuestionとPassageの間にある答えに結び付く情報をAttentionで注視していた．  
提案モデルはquestionの情報を使ってPassageの情報を注視する手法(Self-Matching Attention)を導入した．

# 技術や手法のキモ
out-of-vocabrary(OOB)の対策のために，wordレベルとcharacterレベルのEmbeddingsを使用している．
そして，Answerを出力するのに必要なのはQuestionとAnswer間の情報だけでなくPassage内のContextや語彙の順番といった情報も必要である，と考えPassageの情報を使ってPassageにAttentionをかけるSelf-Matching Attentionを導入したモデルを提案した．


## abstract
文章全体からの情報を効果的にエンコードする、パッセージをそれ自体と照合することによって表現を再調整するための自己マッチングアテンションメカニズムを提案します．

# model
![fig1]  
それに加えて、パッセージ全体からの証拠を集約するために自己マッチング注意を適用し、パッセージ表現を洗練させます。

## Question and Passage Encoder
Question ： $Q=\{w_t^Q\}_{t=1}^m$ , passage : $P=\{w_t^P\}_{t=1}^n$  
word レベルの embedding を $\{e_t^Q\}_{t=1}^m$ , $\{e_t^P\}_{t=1}^n$ とする．  
character レベルの embedding を GRU の最終隠れ状態から取得し， $\{c_t^Q\}_{t=1}^m$ , $\{c_t^P\}_{t=1}^n$ とする．  
新しい表現 $u$ を以下のように取得する．  
$u_t^Q = \text{BiRNN}_Q(u_{t-1}^Q , [e_t^Q,c_t^Q])$  
$u_t^P = \text{BiRNN}_P(u_{t-1}^P , [e_t^P,c_t^P])$

## Gated Attention-based Recurrent Networks
Question 情報を Passage 表現に組み込むために，ゲート付きアテンションベースのリカレントネットワークを提案する．

質問全体 $u^Q$ のアテンションプーリングベクトル $c_t = att(u^Q , [u_t^P,v_{t-1}^P])$ と $u_t^P$ を concat し RNN の入力とし， $v_t^P$ を得る． 

$v_t^P = \text{RNN}(v_{t-1}^P,[u_t^P,c_t])$

パッセージ部分の重要性を判断し，質問に関連する部分に attention するために，RNNの入力（[$u_t^P,c_t$]）にゲートを追加する．

$g_t=\text{sigmoid}(W_g[u_t^P,c_t])$   
$[u_t^P,c_t]^*=g_t\odot [u_t^P,c_t]$

ゲートは文章の一部だけが Question に関連していることを効果的にモデル化する．

## Self-Matching Attention
答えを推測するには、passage の情報が必要．$v^P$をそれ自体に対して直接マッチングする．パッセージ内の単語について、パッセージ全体から証拠を動的に収集し、現在のパッセージ単語に関連する証拠とそれに一致する質問情報をパッセージ表現hPtにエンコードします。

$h_t^P=\text{BiRNN}(h_{t-1}^P,[v_t^P,c_t])$  
$c_t=att(v^P,v_t^P)$

$s_j^t=v^T\tanh(W_v^Pv_j^P+W_v^{\tilde{P}}v_t^P)$  
$a_i^t=\exp(s_i^t)/\sum_{j=1}^n \exp(s_j^t)$  
$c_t=\sum_{i=1}^n a_i^tv_i^P$

## output layer
回答の開始位置と終了位置を予測するために [pointer networks](https://arxiv.org/pdf/1506.03134.pdf) を使用する．加えて pointer network の最初の隠れベクトルを生成するために，質問表現のアテンションプーリングを使用する．  
$\{h_t^P\}_{t=1}^n$ が与えられると，アテンションメカニズムは，開始位置($p^1$)と終了位置($p^2$)を選ぶためのポインタとして用いられ，次のように形式化される．

$s_j^t = \text{v}^T \tanh(W_h^Ph_j^P+W_h^ah_{t-1}^a)$  
$a_i^t=\exp(s_i^t)/\sum_{j=1}^n \exp(s_j^t)$  
$p^t=\text{argmax}(a_1^t,...,a_n^t)$  

ここで $h_{t-1}^a$ は回答のリカレントネットワークの最終隠れ状態を表す．
回答のリカレントネットワークの入力は予測確率 $a^t$ に基づいたアテンションプーリングベクトルである．

$c_t = \sum_{i=1}^n a_i^th_i^P$  
$h_t^a=\text{RNN}(h_{t-1}^a,c_t)$

質問ベクトル $r^Q = att(u^Q,V_r^Q)$ を回答のリカレントネットワークの初期状態として利用する．

Lossは予測された分布によって，真値の開始位置と終了位置の負の対数確率の合計を最小にする．


# 評価実験 
![table2]  
ensembleモデルでは20モデルを使用．  
論文投稿時点のSingleモデルとEnsembleモデルにおいて，Exact Match(EM)/F1の両方で最高のスコアを記録している．

ablationの結果  
![table3]  
self-matching を取り除くと EM が 3.5 減少する．これはパッセージ内の情報が重要な役割を果たすことを示している．特徴レベルの埋め込みは，out-of-vocab やまれな単語をより適切に処理できるため，モデルのパフォーマンスに貢献する．

gating の影響  
![table4]  

attention の可視化結果  
![fig2]  

要素ごとの分析結果  
![fig3]  
Question type ごとに分析すると why のスコアが低い．これはwhyの回答が多様で特定のフレーズに制限されることがないため．

# 議論はあるか？
Self-Matching Attentionのベクトルを可視化すると答えの出力に必要な情報にAttentionがかかっていることが分かった．

# future work
gated attention-based recurrent networks と self-matching attention mechanism を用いた．  
回答の位置予測に pointer-network を用いた．

[MS MARCO](http://www.msmarco.org/) のような他のデータセットで試す．