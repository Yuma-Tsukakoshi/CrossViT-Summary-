# kaggleのデータセット改善点
- [ ] 仮説の検証(仮説ごとにデータを可視化して正当性を理解する)
- [ ] 後処理の実装(post processing, calibrationなど)

## 1 仮説の検証(仮説ごとにデータを可視化して正当性を理解する)
### 1-1 特定の時間帯や曜日に集中して発生する可能性が高い
この仮説を立てたコードを可視化するためのコードを教えてくださいね。このコードは、TransactionDTという特徴量を日時に変換し、年月日や時間などの新しい特徴量を作成しています。また、D9という特徴量に欠損値があるかどうかを示す新しい特徴量も作成しています。

このコードの仮説は、日時や欠損値によって、目的変数であるisFraudの値が変わるというものです。つまり、詐欺行為の発生は、時間帯や曜日などの周期性や、D9という特徴量の有無によって影響されるということです。

この仮説を可視化するためのコードは、以下のようになります。

# 必要なライブラリをインポートする
import matplotlib.pyplot as plt
import seaborn as sns

・ TransactionDTとisFraudの関係を折れ線グラフでプロットする
plt.figure(figsize=(12,6))
plt.title('TransactionDT vs isFraud')
sns.lineplot(x='TransactionDT', y='isFraud', data=train_df)
plt.xlabel('TransactionDT')
plt.ylabel('isFraud')
plt.show()

・ DT_M, DT_W, DT_D, DT_hour, DT_day_week, DT_dayとisFraudの関係を棒グラフでプロットする
cols = ['DT_M', 'DT_W', 'DT_D', 'DT_hour', 'DT_day_week', 'DT_day']
fig, axes = plt.subplots(3, 2, figsize=(12,18))
fig.suptitle('Date and Time Features vs isFraud')
for i, col in enumerate(cols):
    ax = axes[i//2, i%2]
    sns.barplot(x=col, y='isFraud', data=train_df, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel('isFraud')
plt.show()

・ D9とisFraudの関係を棒グラフでプロットする
plt.figure(figsize=(12,6))
plt.title('D9 vs isFraud')
sns.barplot(x='D9', y='isFraud', data=train_df)
plt.xlabel('D9')
plt.ylabel('isFraud')
plt.show()

これらのグラフから、以下のようなことが分かります。

- TransactionDTとisFraudには明確な関係は見られませんが、一定の間隔でisFraudの値が上昇していることがわかります。これは、季節性やイベントなどの要因によるものかもしれません。
- DT_M, DT_W, DT_DとisFraudには弱い正の相関が見られます。これは、時間が経つにつれて詐欺行為が増えていることを示しています。
- DT_hourとisFraudには明確な周期性が見られます。これは、詐欺行為が特定の時間帯に集中していることを示しています。例えば、午前中や深夜に詐欺行為が多く発生しています。
- DT_day_weekとisFraudには弱い負の相関が見られます。これは、詐欺行為が週末に減少する傾向があることを示しています。
- DT_dayとisFraudには明確な関係は見られませんが、月末に詐欺行為が増加する傾向があることがわかります。
- D9とisFraudには強い負の相関が見られます。これは、D9が欠損値である場合に詐欺行為が多く発生することを示しています。

### 1-2 購入者と受取人の一致を確かめる


### 1-3 やりとりに特定のデバイスを用いている



## 2 後処理の実装(post processing, calibrationなど)
Calibrationの方法には、Sigmoid / Platt ScaleやIsotonic Regressionなどがある。  
Sigmoid / Platt Scaleは、モデルの出力値をシグモイド関数にフィットさせて調整する方法です。　　
Isotonic Regressionは、モデルの出力値を単調増加関数にフィットさせて調整する。
LightGBMは、binary log loss classification (or logistic regression) を使っているので、Calibrationは不要だと考えた。

post  processingによって後処理を行う  
予測値の分布を正規化する。  
予測値の平均と標準偏差を計算し、予測値を標準正規分布に変換する。  
⇒予測値のばらつきや外れ値を抑える処理を行う。  

    mean = y.mean() 
    std = y.std()
    
    tt_df = tt_df[['TransactionID',target]] 
    predictions = np.zeros(len(tt_df))
    predictions = (predictions - mean) / std

light_bgmの学習モデルの性能を評価するために、RMSEとR2スコアを出力した。
・post processing後の予測値と実際の値の誤差や精度などの指標を計算し、改善されたかどうかを評価する

    rmse = np.sqrt(mean_squared_error(P_y, predictions)) 
    r2 = r2_score(P_y, predictions) 
    print('RMSE:', rmse) 
    print('R2:', r2)    
    tt_df['prediction'] = predictions
