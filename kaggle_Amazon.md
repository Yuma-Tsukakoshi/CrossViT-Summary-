# kaggleで公式ドキュメント解読しながらsikit_learn実装
  
## 分析コンペにおけるタスクの種類
⇒ 二値分類  
⇒ アクセス権限を与えるかどうか 0/1   

## 評価指標
混同行列 ： 不均衡データの場合は、モデルの性能を評価しづらい　⇒ 分析コンペで使用されるのは少ない  
- 適合率は、予測値に対してどれほど真の値の正例の割合
- 再現率は、真値に対してどれほどどの程度真値予測できてるか  
適合率と再現率はトレードオフの関係にある⇒ F1Scoreによって評価する  
例えば、  
適合率（予測段階でできるだ誤検知を避けたい⇒閾値を緩くする 0.5が基準だが、0.4までを正とする）
⇒ 適合率は上がる ⇔ 真値と外れる可能性が高くなり 再現率が下がる

全体の内の割合とかで考えた場合、明らかに不均衡データだと正例と判断される確率が自然と低くなる
