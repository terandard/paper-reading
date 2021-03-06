Yu Jiang , Vivek Natarajan , Xinlei Chen , Marcus Rohrbach , Dhruv Batra , Devi Parikh  
Facebook AI Research  
[arXiv](https://arxiv.org/abs/1807.09956) , [pdf](https://arxiv.org/pdf/1807.09956.pdf) , [GitHub](https://github.com/facebookresearch/pythia)  

# どんなもの？

# 先行研究との違いは？

## abstract
このドキュメントでは、Facebook AI Research（FAIR）のA-STARチームからVQAチャレンジ2018までの勝利のエントリであるPythia v0.1について説明します。
私たちの出発点は、ボトムアップ トップダウン（アップダウン）モデルの再実装です。 VQA v2.0のデータセット上のアップダウンモデルのパフォーマンスを大幅に向上させることができることを実証しています。これは、モデルアーキテクチャと学習率スケジュールの微妙ではあるが重要な変更を行い、画像の機能を微調整し、 65.67％から70.22％の範囲である。
さらに、異なる特徴と異なるデータセットで訓練されたモデルの多様なアンサンブルを使用することにより、1.31％のアンサンブル（すなわち、異なるランダムシードを有する同じモデル）の「標準的」な方法を大幅に改善することができる。 全体として、VQA v2.0データセットのテスト用分割では72.27％を達成しました。 

## introduction
Pythiaのモチベーションは、今日のVQAモデルの大部分が質問符号化、画像特徴抽出、2つの融合（通常は注意を払う）、回答空間の分類のためのモジュールを備えた特定の設計パラダイムに適合しているという観察から来ている。 Pythiaの長期的な目標は、VQA [2]での簡単でモジュラーな研究開発のためのプラットフォームと、視覚的な対話[3]のような関連する方向の役割を果たすことです。 
Pythia v0.1の出発点は、ボトムアップのトップダウン（up-down）モデル[14]のモジュラー再実装です。 この研究では、一連の微妙で重要な変更を行うことで、表1に要約されているように、パフォーマンスを大幅に改善できることを実証します。

## bottom-up and top-down attention
2017年のVQAチャレンジへの勝利の基盤となったアップダウンモデル[1]のベースラインシステム上でアブレーションと増強を行う。 アップダウンの重要なアイデアは、ボトムアップの注意、すなわちビジュアルフィードフォワードの注意を払って画像の特徴を抽出するために、Visual Genomeデータセット[9]に事前にトレーニングされたより早いRCNN [12]オブジェクト検出器の使用です。 具体的には、ResNet-101をバックボーンネットワークとして選択し、そのRes-5ブロック全体を検出のための第2段階領域分類器として使用した。 訓練の後、各領域は、7×7グリッドから平均プールした後、2048Dの特徴によって表された。
次いで、質問テキストを使用して、画像内の各オブジェクトのトップダウン注意、すなわちタスク特有の注意を計算する。 マルチモーダル融合は、単純なアダマール（Hadamard）プロダクト、続いてシグモイド活性化関数を用いたマルチラベル分類器を用いて行われ、回答スコアを予測する。 彼らのパフォーマンスは、VQA 2.0試験で70.34％に達しました。異なる種子で訓練された30のモデルのアンサンブルでテストを行いました。 説明を分かりやすくするため、提案された変更（およびそれぞれの改善）をシーケンスで提示します。 しかし、我々はまた、それらが独立して有用であることを発見した。

## model Architecture
トレーニングの速度と精度を向上させるため、アップダウンモデルを少し変更しました。 ゲート付き tanh を使用する代わりに、ReLU が後に続くウェイト正規化[13]を使用します。 また、トップダウンのアテンションを計算するときに、フィーチャ連結を要素単位の乗算と置き換えて、テキストと視覚モダリティのフィーチャを結合しました。 質問表現を計算するために、単語埋め込みを初期化するために300D GloVe [11]ベクトルを使用し、次にそれをGRUネットワークに渡し、注意力モジュールを注意深いテキスト特徴[16]を抽出するために渡した。 イメージとテキスト情報を融合させるために、最も優れた隠しサイズが5000であることがわかりました。これらの変更により、VQA v2.0 test-devでモデルのパフォーマンスを65.32％から66.91％に向上させることができました。

## learning schedule
私たちのモデルはAdamaxによって最適化されています.Adamaxは無限のノルムを持つAdamの変種です[8]。 1つの普及した [up-down](https://github.com/hengyuan-hu/bottom-up-attention-vqa) 学習の実装では、バッチサイズを512，学習率0.002に設定されます。バッチサイズを小さくするとパフォーマンスが向上することがわかりました。これは学習率を上げることでパフォーマンスが向上する可能性があることを示しています。 しかし、学習率を単純に増加させると、相違が生じました。 学習率を上げるために、ネットワークの大規模学習率トレーニングに一般的に使用されるウォームアップ戦略[5]を導入しました。 具体的には、0.002の学習率で開始し、iteration が1000，学習率0.01に達するまで各反復で直線的に増加させます。次に、5Kで学習率を0.1倍に減らしてから、2K回の反復ごとに減らし，12Kでトレーニングを停止します。 これにより、test-devのパフォーマンスを66.91％から68.05％に向上させています。

##  Fine-Tuning Bottom-Up Features
事前に訓練されたフィーチャを微調整することは、フィーチャを手元のタスクに合わせて適切に調整してモデルのパフォーマンスを向上させるためのよく知られているテクニックです[12]。
Andersonらとは以下の点で異なる．Detectron の feature pyramid net works（[FPN](https://github.com/facebookresearch/Detectron)) に基づく最新の検出器を使用した．これは ResNeXt [15]をバックボーンとして使用し、領域分類のために2つの完全に接続されたレイヤー（fc6とfc7）を持っています。 これにより、オリジナルのup-down [1]とは対照的に、2048Dのfc6の特徴を抽出し、fc7のパラメータを微調整することができます。前のレイヤーを微調整するためには、7×7×2048の畳み込み特徴マップ上ではるかに多くのストレージ/ IOと計算が必要です。 アップダウンと同様に、私たちは、ビジュアルゲノム（VG）[9]を使用して、オブジェクトと属性の注釈を両方とも検出器を訓練しました。
微調整学習率は全体の学習率の0.1倍に設定しました。 この微調整により、test-devで68.49％のパフォーマンスに達することができます。

## data augument
我々は、Visual Genome [9]およびVisual Dialog（VisDial v0.9）[3]データセットから追加の訓練データを追加した。 VisDialでは、ダイアログ内の10回のターンを10個の独立した質問 - 回答のペアに変換しました。 VQAが10の真値の回答を持つのに対して，VGAとVisDialの両方のデータセットは1つしか持たないため、VGとVisDialの各質問に対する回答を10回複製して、データフォーマットをVQA評価プロトコルと互換性を持たせました。
また、VQAデータセットの画像をミラーリングすることにより、追加のデータ拡大を行った。我々は、それらを含む質問と回答に「左」と「右」のトークンを入れ替えることによって、鏡像の質疑応答を基本的に処理します。これらの追加データセットを追加するときは、最初にセクション2.2で説明したように15K回の反復で学習率を下げ、22K回の反復でトレーニングを停止します。データ増強の結果、シングル・モデルのパフォーマンスを、test-devの68.49％から69.24％に向上させることができます。

## Post-Challenge Improvements
Andersonら [1]は、オブジェクトを提案するボトムアップフィーチャからプールされたフィーチャのみを使用してイメージを表現します。 私たちの仮説は、このような表現は、提案に関するものではない画像領域からの画像および視覚的表現に関する全体的な空間情報を完全には捕捉しないということです。 この仮説を検証するために、グリッドレベルのイメージフィーチャとボトムアップフィーチャを組み合わせました。 我々はResNet152 [7]から格子レベルの特徴を抽出するために[4]と同じ手順に従う。 オブジェクトレベルのフィーチャとグリッドレベルのフィーチャは、質問のフィーチャと別々に融合され、連結されて分類にフィードされます。より包括的な実験を行い、グリッドレベルの機能を追加することでパフォーマンスをさらに69.81％向上させることができました。
[14]で行われたように、画像ごとにオブジェクト提案（10〜100）の数を選択するための適応プロトコルを使用する代わりに、すべての画像に対して100個のオブジェクト提案を使用するより簡単な（しかしより遅い）戦略を試しました。 表1に示すように、バウンディングボックス100個の機能を使用すると、VQA 2.0ではtest-devが70.01％、test-stdが70.24％に達します。

## Model Ensembling
以下に説明するアンサンブル実験には、挑戦期限前に訓練されたモデルが含まれる。 つまり、セクション2.5で説明した2つの挑戦後の実験は含まれていません。 アンサンブルの2つの戦略を試しました。 まず、最高の単一モデルを選択し、同じネットワークを異なるシードでトレーニングし、最終的に各モデルからの予測を平均します。 図1からわかるように、性能は70.96％で安定している。 第2に、データの増強の有無にかかわらずVQAデータセットでトレーニングされた調整されたアップダウンモデルと、データ増強あり/なしの異なるDetectronモデルから抽出された画像フィーチャでトレーニングされたモデルを選択します。 このアンサンブル戦略は、以前のものよりはるかに効果的です。 30種類のモデルを統合すると、VQA v2.0のtest-devで72.18％、test-stdで72.27％に達します。
