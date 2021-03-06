Latent Alignment and Variational Attention  
Yuntian Deng, Yoon Kim, Justin Chiu, Demi Guo, Alexander M. Rush  
[pdf](http://papers.nips.cc/paper/8179-latent-alignment-and-variational-attention.pdf), [GitHub](https://github.com/harvardnlp/var-attn/)  
NIPS2018

# どんなもの？


# 先行研究との差分


# abstract
アテンションネットワークは学習が簡単で alignment をソフトにシミュレートするために効果的な方法  
しかしながら、このアプローチは、確率論においては latent alignments(潜在的な調整) は重要視されていない．however, the approach does not marginalize over latent alignments in a probabilistic sense.   
この性質により、アテンションを他のアラインメント手法と比較すること，それを確率モデルで構成すること，観測データに基づいて事後推論を実行することが困難になる．関連した潜在的なアプローチである hard attention では、これらの問題を解決していますが、一般的に訓練するのは難しく、正確さは劣ります。  
本研究では、latent variable alignment models を学習するためのソフトおよびハードアテンションの代わりに　variational attention networks の検討，および償却された変分推論に基づくより厳密な近似範囲を検討します。with tighter approximation bounds based on amortized variational inference.   
我々はさらに、これらのアプローチを計算的に実行可能にするために勾配の分散を減らすための方法を提案する。実験は、機械翻訳および視覚的質問応答については、非効率的な正確な潜在変数モデルが標準的なneural attentionよりも優れていることを示しているが、ハードアテンションベースのトレーニングを使用するとこれらの利点はなくなる。一方、変分注意はパフォーマンス向上の大部分を保持しますが、トレーニング速度はneural attentionに匹敵します。

# introduction
研究者は、モデルの解釈可能性のためのツールとして、または最終予測の要因として、中間注意の決定を直接使用します。 この観点から、注意は潜在的アラインメント変数の役割を果たす。 もう一つのアプローチであるハードアテンション[80]は、アラインメントのための潜在変数を導入し、それから政策勾配を使って対数限界尤度の限界を最適化することによって、この関係を明確にします。 このアプローチは一般的に（[80]のようないくつかの例外を除いて）パフォーマンスが悪く、ソフトのそれよりも使用頻度が少なくなります。  

それでも、潜在的な位置合わせアプローチは、いくつかの理由で魅力的なままです。
（a）潜在変数は、確率論的な方法で依存関係についての推論を容易にします。例えば 他のモデルとの合成を許可する
（b）事後推論は厳密にフィードフォワードモデルよりもモデル分析と部分予測のためのより良い基礎を提供します。これは機械翻訳のアライメントでは性能が劣ることが示されています[38]。
（c）限界尤度を直接最大化すると、より良い結果が得られる可能性があります。

研究目的は、アテンションを用いて問題を定量化し、変分推論における最近の進展に基づいて代替案を提案することです。 変分推論と厳密な注意との関連は文献[4]、[41]で指摘されていますが、可能な範囲と最適化手法の範囲は十分に検討されておらず、急速に拡大しています。 これらのツールは、ハードアテンションモデルの一般的なパフォーマンスの低さがモデル化の問題（すなわち、ソフトアテンションがより良い帰納的偏りをもたらす）によるものか最適化問題によるものかをより定量化することを可能にする。

私たちの主な貢献は、訓練するために扱いやすいままで潜在的な線形に効果的にフィットすることができる変分注意アプローチです。 私達は変分注意の2つの変種を考えます：カテゴリー的とリラックス。 カテゴリカル法は、学習された推論ネットワークを使用した償却変分推論、およびソフトアテンション分散減少ベースラインを使用したポリシー勾配に適合します。 適切な推論ネットワーク（ソース/ターゲット全体に影響を与える条件）を使用すると、トレーニング時に難しい注意の代わりとして使用することができます。 緩和版は、アライメントがディリクレ分布からサンプリングされ、したがって複数のソース要素にわたる注目を可能にすると仮定する。

実験では、このアプローチを2つの主要な注意ベースのモデルにどのように実装するかを説明しています。それは、神経機械翻訳と視覚質問応答です。 （図1は、機械翻訳のための私たちのアプローチの概要を示しています）。 最初に、厳密な周辺尤度を最大にすると、ソフトアテンションよりもパフォーマンスが向上することを示します。 さらに、変分的（カテゴリカル）アテンションでは、アライメント変数は、それほど難しいトレーニングを必要とせずに、ソフトアテンションとハードアテンションの両方の結果を大幅に上回ることを示します。 我々はさらに、アライメントの決定に対する事後推論の影響、そして潜在変数モデルがどのように採用されるかもしれないかを探求する。

# Variational Attention for Latent Alignment Models
AVIは、学習された推論ネットワークを使用して潜在変数推論を効率的に近似するためのメソッドの一種です。
このセクションでは、ディープ潜在アラインメントモデルのこの手法を探求し、ソフトアテンションとハードアテンションの利点を組み合わせた変分アテンションの方法を提案します。

最初に注意を払う必要があるのは、Jensenの不等式から導き出された下限を最適化することです。 このギャップはかなり大きくなる可能性があり、パフォーマンスの低下につながります。 変分推論法は、このギャップを狭めることを直接目的としています。
特に、evidence lower bound（ELBO）は、分布の族に対するパラメータ化された範囲です。

hard attention は $q(z) = p(z|x,\widetilde{x})$ というELBO の特別な場合

下限証拠を最適化するには多くの方法があります。deep learning では amortized variational inference を使用． AVIは、変分分布 $q(z; \lambda)$ のパラメータを生成するために推論ネットワークを使用します。 推論ネットワークは、入力、クエリ、および出力を取り込みます。 $\lambda = enc(x,\widetilde{x},y;\phi)$
生成モデル $\theta$ を訓練しながら、推論ネットワーク $\phi$ とのギャップを減らすことを目的とする。


最適化戦略および推論ネットワークの正しい選択により、この形態の変分注意は、潜在的アラインメントモデルを学習するための一般的な方法を提供することができる。 このセクションの残りでは、この目的を正確かつ効率的に計算するための戦略について検討します。 次のセクションでは、特定のドメインに対するencのインスタンス化について説明します。

# Algorithm 1: Categorical Alignments
$D$ : the alignment distribution , $Q$ :  the variational family が categorical distribution を考える．  
$\nabla_\phi \text{ELBO}$ は $q(z)$ からの一つのサンプルから簡単に得られる．  
各 $\nabla_\phi \text{ELBO}$ について，ＫＬ部分に関する勾配は容易に計算可能だが，第１項 $\mathbb{E}_{z\sim q(z)}[\log f(x,z)]$ に関する勾配に関する最適化問題がある．  

REINFORCE[76]を特別なベースラインと一緒に使ったアプローチが効果的であることがわかった．ただし、REINFORCEは選択できる推論の選択肢の1つにすぎず、後で説明するように、再パラメータ化可能な緩和などの代替アプローチも機能します。 形式的には、最初に尤度比トリックを適用して、推論ネットワークパラメータφに関する勾配の式を取得します。

$\nabla_\phi \mathbb{E}_{z\sim q(z)}[\log p(y|x,z)] = \mathbb{E}_{z\sim q(z)}[(\log f(x,z)]$

hard attention と同様に single Monte Carlo sample を取得．この推定値の分散減少は $B$．
理想的なベースラインは $\mathbb{E}_{z\sim q(z)}[\log f(x,z)]$ であり，これは強化学習における価値関数に似ている．
この項は簡単には計算できないので，soft attention $\log f(x,\mathbb{E}[z])$ で近似する．
すると勾配は以下の様になる．

$\mathbb{E}_{z\sim q(z)}[(\log \frac{f(x,z)}{f(x,\mathbb{E}_{z'\sim p(z'|x,\widetilde{x})})})\nabla_\phi \log q(z|x,\widetilde{x})]$

これは推論ネットワークアラインメントアプローチとソフトアテンションベースラインとの比に基づいて、ｑへの勾配を重み付けする。 特に、ソフトアテンションにおける期待値は、ｐを超えている（そしてｑを超えていない）ので、ベースラインはφに関して一定である。 同様のベースラインをハードアテンションにも使用できることに注意してください。そして、我々の実験ではそれを変分モデルとハードアテンションモデルの両方に適用します。

# Algorithm 2: Relaxed Alignments
$D$ : the alignment distribution , $Q$ :  the variational family が Dirichlets distribution を考える．  
このモデルは、ある意味では複数のインデックスに質量を割り当てるソフトアテンション定式化に近いものですが、基本的にはアライメントを潜在変数として正式に扱うという点で異なります。 目的は、低分散勾配推定量を見つけることである。 REINFORCEを使用する代わりに、 reparameterization を使用できる特定の連続分布を使用[36]．
ここで、サンプリングｚ〜ｑ（ｚ）は、単純なパラメータ化されていない分布Ｕから最初にサンプリングし、次に変換ｇφ（・）を適用することによって行うことができ、不偏推定量をもたらす。

$\mathbb{E}_{u\sim U}[\nabla_\phi \log p(y|x,g_\phi (u))] - \nabla_\phi \text{KL}[q(z)||p(z|x,\widetilde{x})]$

ディリクレ分布は直接再パラメータ化できません。 標準的な一様分布をディリクレの逆CDFで変換するとディリクレ分布が得られますが、逆CDFには解析解がありません。 ただし、棄却ベースのサンプリングを使用してサンプルを取得し、implicit differentiation を使用してCDFの勾配を推定することができます[32]。

経験的に、我々はランダム初期化がλに対する一様なディリクレパラメータへの収束をもたらすことを見出した。 （シンプレックスの中心に向かって低いKLの局所最適値を見つける方が簡単であると思われます）。 実験では、したがって、最初にJensen bound ($\mathbb{E}_{z\sim p(z|x,\widetilde{x})} [\log p(y|x,z)]$)を最小化することによって潜在的アラインメントモデルを初期化し，推論ネットワークに導入する．


# Models and Methods

## Neural Machine Translation


## VQA

# Experiments
