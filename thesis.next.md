# 追加課題の論文内容

.Cross-Attentionにおける内部の数式の理解
(各層の層数や次元の把握や計算処理)
**https://arxiv.org/pdf/2103.14899.pdf**

![image](https://github.com/Yuma-Tsukakoshi/CrossViT-Summary-/assets/107422037/86926816-82a9-41c5-8377-06a48d7580bc)

・どこの情報が増えると線形になる？  
⇒クロスアテンションモジュールがなぜ用いられているかを調べたところ、二つのブランチ間で情報を交換するために必要な時間と空間の複雑さを線形に抑えることができるため用いられていることがわかりました。　　

線形になるというのは、画像パッチの数に比例して計算量が増えるということです。 通常のTransformerでは、画像パッチの数がNだとすると、Self-Attentionの計算量はO(N2)です。 つまり、画像パッチの数が2倍になると、計算量は4倍になります。しかし、CATでは、画像パッチを特徴マップに分割することで、Self-Attentionの計算量をO(NM)に減らすことができます。 ここで、Mは特徴マップの数です。つまり、画像パッチの数が2倍になっても、計算量は2倍にしかなりません。これは、線形になるということです。線形になるメリットは、より大きな画像やより多くのパッチを扱うことができるということです。

Nをかけられている部分は、画像パッチの数を表しています。画像パッチとは、入力画像を小さな領域に分割したものです。例えば、224×224サイズの画像を16×16サイズのパッチに分割すると、パッチの数は(224/16)2=196になります。この場合、Nは196です。画像パッチの数が多いほど、入力画像の情報量が多くなりますが、計算量も増えます。
Mをかけられている部分は、特徴マップの数を表しています。特徴マップとは、画像パッチを単一チャンネルに分割したものです。例えば、16×16サイズのパッチを4×4サイズの特徴マップに分割すると、特徴マップの数は(16/4)2=16になります。この場合、Mは16です。特徴マップの数が多いほど、画像パッチ内部の情報量が多くなりますが、計算量も増えます。
CATでは、NとMを調整することで、画像パッチ内部と外部の両方にAttentionを適用することができます。 Attentionとは、queryとmemory（keyとvalue）という二つの要素からなるメカニズムで、queryによってmemoryから必要な情報を選択的に取り出すことができます。 CATでは、以下のような数式でCross Attentionを表現します。
CAT(Q,K,V)=softmax(dk​​SA(Q)SA(K)⊤​)SA(V)
ここで、SAはSelf-Attentionを表す関数で、
SA(X)=softmax(dx​​XX⊤​)X
です。Xは任意の行列であり、dx​はその次元数です。この数式は、Self-Attentionしたqueryとkeyの類似度（内積）を計算し、それを正規化して重み付けしたSelf-Attentionしたvalueの和を求めることを意味します。
Cross Attentionでは、queryはDecoderから、memory（keyとvalue）はEncoderから与えられます。 つまり、
CrossAttention(Q,K,V)=softmax(dk​​QK⊤​)V
ここで、QはDecoderの出力行列、KとVはEncoderの出力行列です。この数式は、Decoderの出力がEncoderの出力にどれだけ関連しているかを計算し、それに応じてEncoderの出力から情報を取り出すことを意味します。
CATでは、Cross Attentionを拡張して、画像パッチの内部と外部にAttentionを適用します。 まず、画像パッチを単一チャンネルの特徴マップに分割し、それぞれに対してSelf-Attention（queryとmemoryが同じ場合のAttention）を適用します。これにより、画像パッチ内部の局所的な情報が捉えられます。次に、特徴マップ間でCross Attentionを適用します。これにより、画像パッチ外部の大域的な情報が捉えられます。


・各層の次元数の把握 
・数式を深く理解する 　　
Transformerは、EncoderとDecoderからなるモデルで、Encoderは入力画像をパッチに分割し、それぞれに位置エンコーディングを加えて特徴ベクトルに変換します。Decoderは、Encoderから受け取った特徴ベクトルと自身の出力を用いて、最終的な分類結果を生成します。1
Attentionは、queryとmemory（keyとvalue）という二つの要素からなるメカニズムで、queryによってmemoryから必要な情報を選択的に取り出すことができます。2 Attentionは、以下のような数式で表されます。
Attention(Q,K,V)=softmax(dk​​QK⊤​)V
ここで、Qはqueryの行列、Kはkeyの行列、Vはvalueの行列、dk​はkeyの次元数です。softmaxは行ごとに正規化する関数です。この数式は、queryとkeyの類似度（内積）を計算し、それを正規化して重み付けしたvalueの和を求めることを意味します。
Cross Attentionでは、queryはDecoderから、memory（keyとvalue）はEncoderから与えられます。1 つまり、
CrossAttention(Q,K,V)=softmax(dk​​QK⊤​)V
ここで、QはDecoderの出力行列、KとVはEncoderの出力行列です。この数式は、Decoderの出力がEncoderの出力にどれだけ関連しているかを計算し、それに応じてEncoderの出力から情報を取り出すことを意味します。
CATでは、Cross Attentionを拡張して、画像パッチの内部と外部にAttentionを適用します。4 まず、画像パッチを単一チャンネルの特徴マップに分割し、それぞれに対してSelf-Attention（queryとmemoryが同じ場合のAttention）を適用します。これにより、画像パッチ内部の局所的な情報が捉えられます。次に、特徴マップ間でCross Attentionを適用します。これにより、画像パッチ外部の大域的な情報が捉えられます。
CATでは、以下のような数式でCross Attentionを表現します。
CAT(Q,K,V)=softmax(dk​​SA(Q)SA(K)⊤​)SA(V)
ここで、SAはSelf-Attentionを表す関数で、
SA(X)=softmax(dx​​XX⊤​)X
です。Xは任意の行列であり、dx​はその次元数です。この数式は、Self-Attentionしたqueryとkeyの類似度（内積）を計算し、それを正規化して重み付けしたSelf-Attentionしたvalueの和を求めることを意味します。
CATのモデルの構造は、以下の図4に示されています。EncoderとDecoderは、それぞれ12層のTransformerブロックからなります。各Transformerブロックは、Self-Attention、Cross Attention、Feed Forward Network（全結合層）からなります。各Attention層の出力次元はD=768です。画像パッチのサイズはP=16×16で、特徴マップのサイズはM=4×4です。つまり、各画像パッチはM2=16個の特徴マップに分割されます。最終的な分類層は、Decoderの最初の位置にある特殊なトークン（[CLS]）に対応する出力を用います
