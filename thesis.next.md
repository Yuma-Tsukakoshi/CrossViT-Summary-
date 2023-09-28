# 追加課題の論文内容

.Cross-Attentionにおける内部の数式の理解
(各層の層数や次元の把握や計算処理)  
**https://arxiv.org/pdf/2103.14899.pdf**

![image](https://github.com/Yuma-Tsukakoshi/CrossViT-Summary-/assets/107422037/86926816-82a9-41c5-8377-06a48d7580bc)

・前回いただいた追加課題を説明する前に一通りモデルの概要を説明します。  

【上記の画像の説明を大枠を説明する。】

・説明が終わったところで前回いただいた追加課題は、  
内部の数式の理解と次元数の把握を行うことでした。また、crossViTにおいて計算量が削減できる理由に計算量が線形になることを挙げたと思うのですがより詳しく話せたらと思います。

・線形になってより計算量が減ったとは？？  
線形になるというのは、画像パッチの数に比例して計算量が増えるということです。 通常のTransformerでは、画像パッチの数がNだとすると、Self-Attentionの計算量はO(N2)です。 つまり、画像パッチの数が2倍になると、計算量は4倍になります。しかし、CATでは、画像パッチを特徴マップに分割することで、Self-Attentionの計算量をO(NM)に減らすことができます。 ここで、Mは特徴マップの数です。つまり、画像パッチの数が2倍になっても、計算量は2倍にしかなりません。これは、線形になるということです。線形になるメリットは、より大きな画像やより多くのパッチを扱うことができるということです。

Nをかけられている部分は、画像パッチの数を表しています。画像パッチとは、入力画像を小さな領域に分割したものです。例えば、$224\times 224$サイズの画像を$16\times 16$サイズのパッチに分割すると、パッチの数は$(224/16)^2=196$になります。この場合、Nは196です。画像パッチの数が多いほど、入力画像の情報量が多くなりますが、計算量も増えます。

Mをかけられている部分は、特徴マップの数を表しています。特徴マップとは、画像パッチを単一チャンネルに分割したものです。例えば、$16\times 16$サイズのパッチを$4\times 4$サイズの特徴マップに分割すると、特徴マップの数は$(16/4)^2=16$になります。この場合、Mは16です。特徴マップの数が多いほど、画像パッチ内部の情報量が多くなりますが、計算量も増えます。

CATでは、NとMを調整することで、画像パッチ内部と外部の両方にAttentionを適用することができます。 Attentionとは、queryとmemory（keyとvalue）という二つの要素からなるメカニズムで、queryによってmemoryから必要な情報を選択的に取り出すことができます。 CATでは、以下のような数式でCross Attentionを表現します。

$$
\mathrm{CAT}(Q, K, V) = \mathrm{softmax}(\frac{\mathrm{SA}(Q)\mathrm{SA}(K)^\top}{\sqrt{d_k}})\mathrm{SA}(V)
$$

ここで、$\mathrm{SA}$はSelf-Attentionを表す関数で、

$$
\mathrm{SA}(X) = \mathrm{softmax}(\frac{XX^\top}{\sqrt{d_x}})X
$$

です。$X$は任意の行列であり、$d_x$はその次元数です。この数式は、Self-Attentionしたqueryとkeyの類似度（内積）を計算し、それを正規化して重み付けしたSelf-Attentionしたvalueの和を求めることを意味します。

Cross Attentionでは、queryはDecoderから、memory（keyとvalue）はEncoderから与えられます。 つまり、

$$
\mathrm{CrossAttention}(Q, K, V) = \mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}})V
$$

ここで、$Q$はDecoderの出力行列、$K$と$V$はEncoderの出力行列です。この数式は、Decoderの出力がEncoderの出力にどれだけ関連しているかを計算し、それに応じてEncoderの出力から情報を取り出すことを意味します。

CATでは、Cross Attentionを拡張して、画像パッチの内部と外部にAttentionを適用します。 まず、画像パッチを単一チャンネルの特徴マップに分割し、それぞれに対してSelf-Attention（queryとmemoryが同じ場合のAttention）を適用します。これにより、画像パッチ内部の局所的な情報が捉えられます。次に、特徴マップ間でCross Attentionを適用します。これにより、画像パッチ外部の大域的な情報が捉えられます。


・各層の次元数の把握 
・数式を深く理解する 　　
この論文では、画像分類のためのTransformerベースのモデルであるCAT（Cross Attention in Vision Transformer）を提案しています。CATは、画像パッチの内部と外部の両方にAttentionを適用することで、局所的な情報と大域的な情報を同時に捉えることができます。Cross Attentionの内部の数式の理解には、まずTransformerの基本的な構造とAttentionの仕組みについて知っておく必要があります。¹²³

Transformerは、EncoderとDecoderからなるモデルで、Encoderは入力画像をパッチに分割し、それぞれに位置エンコーディングを加えて特徴ベクトルに変換します。Decoderは、Encoderから受け取った特徴ベクトルと自身の出力を用いて、最終的な分類結果を生成します。¹

Attentionは、queryとmemory（keyとvalue）という二つの要素からなるメカニズムで、queryによってmemoryから必要な情報を選択的に取り出すことができます。² Attentionは、以下のような数式で表されます。

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}})V
$$

ここで、$Q$はqueryの行列、$K$はkeyの行列、$V$はvalueの行列、$d_k$はkeyの次元数です。$\mathrm{softmax}$は行ごとに正規化する関数です。この数式は、queryとkeyの類似度（内積）を計算し、それを正規化して重み付けしたvalueの和を求めることを意味します。

Cross Attentionでは、queryはDecoderから、memory（keyとvalue）はEncoderから与えられます。¹ つまり、

$$
\mathrm{CrossAttention}(Q, K, V) = \mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}})V
$$

ここで、$Q$はDecoderの出力行列、$K$と$V$はEncoderの出力行列です。この数式は、Decoderの出力がEncoderの出力にどれだけ関連しているかを計算し、それに応じてEncoderの出力から情報を取り出すことを意味します。

CATでは、Cross Attentionを拡張して、画像パッチの内部と外部にAttentionを適用します。⁴ まず、画像パッチを単一チャンネルの特徴マップに分割し、それぞれに対してSelf-Attention（queryとmemoryが同じ場合のAttention）を適用します。これにより、画像パッチ内部の局所的な情報が捉えられます。次に、特徴マップ間でCross Attentionを適用します。これにより、画像パッチ外部の大域的な情報が捉えられます。

CATでは、以下のような数式でCross Attentionを表現します。

$$
\mathrm{CAT}(Q, K, V) = \mathrm{softmax}(\frac{\mathrm{SA}(Q)\mathrm{SA}(K)^\top}{\sqrt{d_k}})\mathrm{SA}(V)
$$

ここで、$\mathrm{SA}$はSelf-Attentionを表す関数で、

$$
\mathrm{SA}(X) = \mathrm{softmax}(\frac{XX^\top}{\sqrt{d_x}})X
$$

です。$X$は任意の行列であり、$d_x$はその次元数です。この数式は、Self-Attentionしたqueryとkeyの類似度（内積）を計算し、それを正規化して重み付けしたSelf-Attentionしたvalueの和を求めることを意味します。

CATのモデルの構造は、以下の図⁴に示されています。EncoderとDecoderは、それぞれ12層のTransformerブロックからなります。各Transformerブロックは、Self-Attention、Cross Attention、Feed Forward Network（全結合層）からなります。各Attention層の出力次元は$D=768$です。画像パッチのサイズは$P=16\times 16$で、特徴マップのサイズは$M=4\times 4$です。つまり、各画像パッチは$M^2=16$個の特徴マップに分割されます。最終的な分類層は、Decoderの最初の位置にある特殊なトークン（[CLS]）に対応する出力を用います。
