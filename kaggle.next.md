# kaggleのデータセット改善点
- [x] 仮説の検証(仮説ごとにデータを可視化して正当性を理解する)
- [x] 後処理の実装(post processing, calibrationなど)

## 1 仮説の検証(仮説ごとにデータを可視化して正当性を理解する)
### 1-1 特定の時間帯や曜日に集中して発生する可能性が高い
このコードの仮説は、日時や欠損値によって、目的変数であるisFraudの値が変わるというものです。つまり、詐欺行為の発生は、時間帯や曜日などの周期性や、D9という特徴量の有無によって影響されるということです。

この仮説を可視化するためのコードは、以下のようになります。

必要なライブラリをインポートする
```python
import seaborn as sns
import matplotlib.pyplot as plt
```

DT_hour, DT_day_week, DT_dayとisFraudの関係を棒グラフでプロットする
```python
cols = ['DT_hour', 'DT_day_week', 'DT_day']
fig, axes = plt.subplots(3, 2, figsize=(12,18))
fig.suptitle('Date and Time Features vs isFraud')
for i, col in enumerate(cols):
    ax = axes[i//2, i%2]
    sns.barplot(x=col, y='isFraud', data=train_df, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel('isFraud')
plt.show()
```

D9とisFraudの関係を棒グラフでプロットする
```python
plt.figure(figsize=(12,6))
plt.title('D9 vs isFraud')
sns.barplot(x='D9', y='isFraud', data=train_df)
plt.xlabel('D9')
plt.ylabel('isFraud')
plt.show()
```

これらのグラフから、わかったこと。

- DT_hourとisFraudには明確な周期性が見られます。これは、詐欺行為が特定の時間帯に集中していることを示しています。例えば、午前中や深夜に詐欺行為が多く発生しています。
- DT_day_weekとisFraudには弱い負の相関が見られます。これは、詐欺行為が週末に減少する傾向があることを示しています。
- DT_dayとisFraudには明確な関係は見られませんが、月末に詐欺行為が増加する傾向があることがわかります。
- D9とisFraudには強い負の相関が見られます。これは、D9が欠損値である場合に詐欺行為が多く発生することを示しています。

総じて、不正取引は特定の時間帯に行われていることがわかった。
１週間で見ると特に差は見られなかったが１日単位で見ると、早朝や深夜帯が狙われやすいことがわかった。そして、月末に件数が増えることもわかった。
従って、特定の時間帯に不正取引が行われているという仮説はおおむね的を射ているのかと感じる。

### 1-2 購入者と受取人の一致を確かめる
P_emaildomainとR_emaildomainという特徴量を使って、購入者と受取人のメールアドレスの一致や欠損値の有無を調べています。また、email_checkという新しい特徴量を作成して、メールアドレスが一致する場合は1、不一致や欠損値の場合は0としています。

このコードの仮説は、メールアドレスの一致や欠損値によって、目的変数であるisFraudの値が変わるというものです。つまり、詐欺行為の発生は、購入者と受取人の関係や信頼性によって影響されるということです。

この仮説を可視化するためのコードは、以下のようになります。

必要なライブラリをインポートする
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

P_emaildomainとisFraudの関係をカウントプロットでプロットする
```python
plt.figure(figsize=(12,6))
plt.title('P_emaildomain vs isFraud')
sns.countplot(x='P_emaildomain', hue='isFraud', data=train_df)
plt.xlabel('P_emaildomain')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```

R_emaildomainとisFraudの関係をカウントプロットでプロットする
```python
plt.figure(figsize=(12,6))
plt.title('R_emaildomain vs isFraud')
sns.countplot(x='R_emaildomain', hue='isFraud', data=train_df)
plt.xlabel('R_emaildomain')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
```

email_checkとisFraudの関係をカウントプロットでプロットする
```python
plt.figure(figsize=(12,6))
plt.title('email_check vs isFraud')
sns.countplot(x='email_check', hue='isFraud', data=train_df)
plt.xlabel('email_check')
plt.ylabel('Count')
plt.show()
```

これらのグラフから、以下のようなことが分かります。

- P_emaildomainとisFraudには明確な関係は見られませんが、gmail.comやyahoo.comなどの一般的なメールアドレスが多く使われていることがわかります。
- R_emaildomainとisFraudには強い正の相関が見られます。これは、受取人のメールアドレスがgmail.comやhotmail.comなどの一般的なメールアドレスである場合に詐欺行為が多く発生することを示しています。
- email_checkとisFraudには相関関係が見られなかった。仮説で取り上げたメールアドレスが一致するケースは詐欺にあまり関係ないことがわかった。
根拠としては⇒email_checkでメアドの相違に関わらす不正取引件数は変化がないのと、逆にメアドが違う方が不正取引ではない件数が多いことから
仮説は間違っていたと判断できる。

### 1-3 やりとりに特定のデバイスを用いている
train_identityとtest_identityというデータフレームに対して、DeviceInfo, id_30, id_31という特徴量を使って、デバイスやブラウザの種類やバージョンを抽出しています。また、新しい特徴量として、DeviceInfo_device, DeviceInfo_version, id_30_device, id_30_version, id_31_deviceという列を作成しています。

このコードの仮説は、デバイスやブラウザの種類やバージョンによって、目的変数であるisFraudの値が変わるというものです。つまり、詐欺行為の発生は、ユーザーが使用する機器やソフトウェアによって影響されるということです。

必要なライブラリをインポートする
```python
import matplotlib.pyplot as plt
import seaborn as sns
```
id_30_deviceとisFraudの関係をカウントプロットでプロットする
```python
plt.figure(figsize=(12,6))
plt.title(‘id_30_device vs isFraud’)
sns.countplot(x=‘id_30_device’, hue=‘isFraud’, data=train_df)
plt.xlabel(‘id_30_device’)
plt.ylabel(‘Count’)
plt.xticks(rotation=90)
plt.show()
```
id_31_deviceとisFraudの関係をカウントプロットでプロットする
```python
plt.figure(figsize=(12,6))
plt.title(‘id_31_device vs isFraud’)
sns.countplot(x=‘id_31_device’, hue=‘isFraud’, data=train_df)
plt.xlabel(‘id_31_device’)
plt.ylabel(‘Count’)
plt.xticks(rotation=90)
plt.show()
```

これらのグラフから、以下のようなことが分かります。

id_30_deviceとisFraudには強い正の相関が見られます。これは、OSの種類によって詐欺行為が多く発生することを示しています。例えば、androidやiosなどのモバイルOSで詐欺行為が多く発生しています。また、欠損値でデバイスの情報がないものに関しても不正取引が多いことがわかる。

id_31_deviceとisFraudには強い正の相関が見られます。これは、ブラウザの種類によって詐欺行為が多く発生することを示しています。例えば、chromeやsafariなどの一般的なブラウザで詐欺行為が多く発生しています。

総じて、特定のデバイスやなじみのある機器を用いて詐欺行為が行われている。つまり、特定のデバイスを用いて不正取引が起こるという仮説はある程度適切だったといえると思います。

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
