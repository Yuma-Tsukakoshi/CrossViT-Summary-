# 追加課題
### 1-1.一つ一つ処理を施し精度の結果を見る
カラムのつけ外しを行い、どのカラムが本当に精度に影響しているかを確かめる。
 
### 1-2.仮説2,3に関してのデータの検証方法
不正利用を件数で見るのではなく割合で可視化する。  
kaggleのページ見せる⇒カウントの時と全く異なる形でグラフを可視化することができた。

### 1-3.ROCの定義を確認する
後処理を行う前と後では、多少の誤差はあったのですが、結果が変わらないことを確認できました。  
そこでROCがどのような計算方法になっているかを確認しました。  
平均値と分散が影響しないのは、ROC曲線やAUCは確率の値に基づいて計算されるからです。確率の値は、0から1までの範囲に収まります。  
平均値と分散を使って正規化すると、データのスケールが変わりますが、確率の値には影響しない。  

平均値と分散を使って正規化すると、データの分布が変わります。データの分布が変わると、分類器の学習に影響を与える可能性があります。例えば、データの分布が正規分布に近づくと、分類器の学習が容易になる場合があります。しかし、分類器の評価には影響しません。分類器の評価は、分類器の出力を確率として扱い、閾値を変えることで計算します。

データのスケールが変わっても確率は変わらないのは、確率はデータの相対的な頻度を表すからです。例えば、あるクラスのテストの点数が正規分布に従っているとします。
そのとき、点数を10倍にしたり、平均を引いたりしても、各点数が出る確率は変わりません。なぜなら、データのスケールを変えても、データの順序や分布は変わらないからです。

![image](https://github.com/Yuma-Tsukakoshi/CrossViT-Summary-/assets/107422037/9d377c4e-35f0-471b-9c0d-c25c21172b01)

![image](https://github.com/Yuma-Tsukakoshi/CrossViT-Summary-/assets/107422037/fd5f4572-0001-46e4-b32d-3a7aadda40c8)