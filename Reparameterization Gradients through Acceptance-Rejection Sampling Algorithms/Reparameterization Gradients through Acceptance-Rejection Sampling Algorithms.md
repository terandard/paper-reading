Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms を読んだ
Christian A. Naesseth, Francisco J. R. Ruiz, Scott W. Linderman, David M. Blei  
[pdf](http://www.cs.columbia.edu/~blei/papers/NaessethRuizLindermanBlei2017.pdf), [GitHub](https://github.com/blei-lab/ars-reparameterization)  
Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)2017

# どんなもの？


# 先行研究との差分


# abstract
reparameterization trick を使用した変分推論は、複雑な確率モデルにおける大規模な近似ベイズ推定を可能にし、確率的な最適化を利用して手に負えない予想を回避しました。 reparameterization trick は、分布が固定されている補助確率変数に微分可能な決定論的関数を適用することによって確率変数をシミュレートできる場合に適用できます。多くの関心のある分布（ガンマやディリクレなど）では、確率変数のシミュレーションは合否判定サンプリング(acceptance-rejection sampling)に依存します。 受け入れ - 拒否のステップによって生じる不連続性は、標準的なreparameterization trick が適用されないことを意味します。 変数がacceptance-rejection samplingアルゴリズムの出力である場合でも、再パラメータ化勾配を利用できるようにする新しい方法を提案します。 我々のアプローチは、より大きなクラスの変分分布の再パラメータ化を可能にします。 実データおよび合成データに関するいくつかの研究では、勾配の推定量の分散が他の最先端の方法よりも有意に低いことを示しています。 これは確率的勾配変分推論のより速い収束を導く。

# introduction
変分推論(Variational inference)は、大規模な確率論的モデリングにおける最近の多くの進歩の根底にあります。 画像やテキストなどの複雑なドメインの高度なモデリングが可能になりました。
定義上、変分アプローチの成功は、以下の能力にかかっています。
（i）柔軟なパラメトリック分布族(family)を定式化する。
（ii）パラメーターを最適化して、真の事後分布に最も近いこのファミリーのメンバーを見つけます。
これら2つの基準は矛盾しています - ファミリがより柔軟であるほど、最適化問題はより困難になります。 本論文では、大クラスの変分分布、すなわち acceptance-rejection sampling または rejection sampling によって効率的にシミュレートできる分布に対して、より効率的な最適化を可能にする新しい方法を提示する．

複雑なモデルの場合、変分パラメータは、evidence lower bound (elbo，証拠の下限)やデータの周辺尤度(marginal likelihood)の下限に基づいて確率勾配を上昇させることによって最適化できます。 elboの勾配を推定するには、2つの主要な手段があります。score function estimator と reparameterization trickです。どちらもモンテカルロサンプリングに依存します。 reparameterization trickは、より低い分散推定値をもたらし、効率的な最適化をもたらすが、このアプローチは、その範囲が少数の変分族（典型的にはガウス分布）に限られていた。 いくつかの研究がすでにこの制限に対処しようとしています。

reparameterization trickを適用するには2つの要件があります。
1つ目は、確率変数は、uniform またはstandard normalなどの単純な確率変数の変換によって取得できることです。
2つ目は、変換が区別可能(differentiable)であるということです。
この論文では、私たちがコンピューター上でシミュレートするすべての確率変数は、最終的には uniform の変換であり、その後にaccept-reject stepが続くことがわかります。 そのため、変換が微分可能であれば、これらの既存のシミュレーションアルゴリズムを使用して reparameterization trick の範囲を拡大できます。

このように、変分パラメータの確率的勾配を作成するために既存の rejection sampler を使用する方法を示します。 つまり、各 rejection sampler は、その分布に適した高度に調整された変換を使用します。 これらのブラックボックスから「ふたを外す」ことによって、新しいreparameterization gradient を構築でき、変分推論への変換に関する65年以上の研究を適用することができます。 我々はこれが効率的な推論に従う変分モデルの範囲を広げ、最先端の手法と比較して勾配の低分散推定値を提供することを実証する。

確率勾配法に焦点を当てて、まず変分推論をレビューします。 次に、私たちの重要な貢献であるrejection sampling variational inference（rsvi）を提示し、変分目的の低分散確率勾配を生成するための効率的な棄却サンプラーの使用方法を示します。 ガンマとディリクレの棄却サンプラーを分析して、対応する変分要因に対する新しい再パラメータ化勾配を作成します。 最後に、rsviと最先端技術を比較しながら、deep exponential family（def）を持つ2つのデータセットを分析します。 我々は、rsviが大幅な分散の減少とelboのより速い収束を達成することを発見しました。


# Reparameterizing the Acceptance-Rejection Sampler
reparameterization の基本的な考え方は、複雑な分布からそのパラメータとより単純な確率変数のセットの決定論的マッピングとしてシミュレーションを書き直すことです。 rejection samplerは、uniforms, normalsなどの単純な乱数の複雑な決定論的マッピングとして見ることができます。 このため、棄却サンプラーによって生成された確率変数を検討するときは、標準的な再パラメータ化アプローチを採用するのは魅力的です。 ただし、このマッピングは一般に連続的ではないため、導関数を期待内に移動して直接自動微分を使用しても必ずしも正しい答えが得られるとは限りません。

我々の洞察は、我々が代わりに容認されたサンプルに対する限界だけを考えることによってこの問題を克服することができるということです、accept-reject変数を分析的に統合します。 したがって、マッピングは提案ステップから来ます。 これは穏やかな仮定の下では連続的であり、再パラメータ化に適した変分族のクラスを大幅に拡張することができます。

我々は最初に拒絶サンプリングを検討し、そして再パラメータ化された拒絶サンプラを提示する。 次に、それを使ってエルボの低分散勾配を計算する方法を示します。 最後に、変分推論rsviに対する完全な確率的最適化を示します。

## Reparameterized Rejection Sampling
Acceptance-Rejection samplingは、逆累積分布関数が利用できない、またはDevroyeを評価するには高すぎる複雑な分布から確率変数をシミュレートする強力な方法です。 我々は、リパラメータ化トリックを明示的に利用する、拒絶サンプリングの代替的な見方を検討する。 この棄却サンプラーの見方は、3.2節で変分推論アルゴリズムを可能にします。

rejection sampling を用いて分布 $q(z;\theta)$ からサンプルを生成するために，まず proposal distribution $r(z;\theta)$ からサンプルする． ただし $q(z;\theta)<=M_\theta r(z;\theta)$ , $M_\theta < inf$  
rejection sampler では proposal distribution が再パラメータ化可能であると仮定，すなわち $z \sim r(z;\theta)$ を生成することは，$\varepsilon \sim s(\varepsilon)$ を生成し，微分可能関数 $h(\varepsilon),\theta)$ を用いて $z=h(\varepsilon,\theta)$ とすることと同等である．  
次に以下の確率で accept し，それ以外の場合は、サンプルを reject してプロセスを繰り返す

$min\{1,\frac{q(h(\varepsilon,\theta);\theta)}{M_\theta r(h(\varepsilon,\theta);\theta)}\}$

![fig1]

![algorithm1]  

$h(\varepsilon,\theta)$ を介した再パラメータ化によって $r(z;\theta)$ からシミュレートする能力は、拒絶サンプラが有効であるためには必要ではない。 しかしながら、これは実際に多くの一般的な分布のリジェクションサンプラに当てはまります。


## The Reparameterized Rejection Sampler in Variational Inference

elbo の勾配の新しいモンテカルロ推定量を開発するために、再パラメータ化された棄却サンプリングを使用します。 最初に、変換された変数εに関する期待値として（1）のエルボを書き換えます。


$$
\begin{align}
 L(\theta) &=  \mathbb{E}_{q(z;\theta)}[f(z)]+\mathbb{H}[q(z;\theta)] \\\
   &= \mathbb{E}_{\pi(\varepsilon;\theta)}[f(h(\varepsilon),\theta))]+\mathbb{H}[q(z;\theta)]
\end{align}
$$

$\pi(\varepsilon;\theta)$ はAlgorithm 1 で accept されたサンプル $\varepsilon$ の分布  
補助一様変数uをmarginalizingして構築


$$
\begin{align}
 \pi(\varepsilon;\theta) &= \int \pi(\varepsilon,u;\theta)du \\\
 &= \int M_\theta s(\varepsilon)\mathbb{1}[0 < u < \frac{q(h(\varepsilon,\theta);\theta)}{M_\theta r(h(\varepsilon,\theta);\theta)}]du \\\
 &= s(\varepsilon)\frac{q(h(\varepsilon,\theta);\theta)}{r(h(\varepsilon,\theta);\theta)}
\end{align}
$$

$\mathbb{1}[x\in A]$ は indicator function, $M_\theta$ は rejection sampler で使用される定数  

### Proposition 1. 
$f$ を計測可能な関数，$\varepsilon \sim \pi(\varepsilon;\theta)$ のとき，  

![pro1]

$\mathbb{E}_{q(z;\theta)}[f(z)]$ の勾配を次の様に計算できる  
![eq5]  

log-derivative trick を使用し，積分を $\pi(\varepsilon;\theta)$ の期待値として書き換える．  
モデルとその潜在変数に関する勾配を利用する再パラメータ化項として $g_{rep}$ を定義．  
$r(z;\theta) \equiv q(z;\theta)$ を使用しないことを説明する補正項として $g_{cor}$ を定義

式5を用いて式1のELBOの勾配は以下の様に表す  
![eq6]

unbiased one-sample Monte Carlo estimator $\hat g \approx \nabla_\theta L(\theta)$ を以下の様に構築する  
![eq7]  

より多くのサンプルを生成することができるが，実際には単一のサンプルで十分であった．  

$h(\varepsilon,\theta)$ が $\varepsilon$ で可逆であるならば、$g_{cor}$ での対数比の勾配の評価を単純化することができる  
![eq8]

また，勾配を $s(\varepsilon)$ に関する期待値として書き換えることができる  
![eq]  
importance sampling-based Monte Carlo estimator を作成．ここで重要度の重みは $q(h(\varepsilon,\theta ) ; \theta) /r (h(\varepsilon, \theta)$ になる．しかし，高次元では重要度の重みの分散が大きすぎるため，このアプローチは低次元の問題に対してのみ有益であると予想．

## Full Algorithm
elbo の勾配のモンテカルロ推定量を得るために式６を利用．  
確率的勾配ステップをとるためにこの推定量を使用  
Kucukelbirらによって提案された step-size sequence $\rho^n$ を使用  
![eq9]  
$n$ はイテレーション数，$\delta=10^{-16}$，$t=0.1$とする．

![alg2]

# 評価実験
## 関連研究
automatic differentiation variational inference (advi)[Kucukelbir et al., 2015, 2016]
adviは、確率変数が実数上にあるように確率変数に変換を適用してから、変換された変数εに対してガウス変分事後近似を配置します。 このように、adviは標準的な再パラメータ化を可能にしますが、例えば、ガンマやDirichletの変分後部には適しません。 したがって、ルイスらによって指摘されているように、ａｄｖｉは確率密度を特異点で近似するのに苦労している。 [2016年]。 これとは対照的に、我々のアプローチは、我々がより広いクラスの変分分布に再パラメータ化トリックを適用することを可能にし、それは正確な事後がまばらさを示すときにより適切であるかもしれない。

別の研究分野では、離散空間の連続的緩和を介して離散潜在変数モデルに再パラメータ化を適用することに焦点が当てられています。 [Maddison
et al., 2017, Jang et al., 2017] 

The generalized reparameterization (g-rep) method[Ruiz et al., 2016]
十分なzの統計量の標準化に基づく変換を適用することによって、勾配の decomposition をgrep + gcorとして利用します。
我々のアプローチはg-repとは異なり、εの分布を変分パラメータに弱く依存させるzの変換を探すのではなく（すなわち、標準化）、
z = h（ε、θ）の分布がq（z;θ）にほぼ等しくなるように、単純な確率変数εの変換を選択することによって、反対のことを行います。
そのために、我々は拒絶サンプリングで通常使用される変換を再利用します。 変分分布ごとに新しい変換を導き出す必要があるのではなく、棄却サンプリングの文献[Devroye、1986]の変換に関する何十年もの研究を活用します。 棄却サンプリングでは、これらの変換（およびεの分布）は、高い受け入れ確率を持つように選択されます。つまり、rsviを使用してgcor≈0を取得することを期待する必要があります。

最後に、非共役変分推論における別の研究は、より表現力のある変分族を開発することを目的としています。 棄却サンプリングが確率変数を生成するために使用されるときはいつでも、rsviはこれらの方法に同様に再パラメータ化トリックを拡張することができます。


## Examples of Acceptance-Rejection Reparameterization

### ガンマ分布
$\alpha\geq1$のとき $Gamma(\alpha,1)$ について Marsaglia and Tsang は効率的な rejection sampler を開発した．  
![eq10]  
$\beta\neq1$ のとき，$z/\beta$ によってサンプルが得られる．

$\alpha < 1$ のとき，$Gamma(\alpha,\beta)$ について $\tilde{z}\sim Gamma(\alpha+1,\beta)$ , $u\sim U[0,1]$ とし，$z=u^{1/\alpha}\tilde{z}$ を得る．  


![fig2]

### ディリクレ分布
