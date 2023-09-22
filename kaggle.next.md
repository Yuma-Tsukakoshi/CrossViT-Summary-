# kaggleのデータセット改善点
- [ ] 仮説の検証(仮説ごとにデータを可視化して正当性を理解する)
- [ ] 後処理の実装(post processing, calibrationなど)

## 1 仮説の検証(仮説ごとにデータを可視化して正当性を理解する)


## 2 後処理の実装(post processing, calibrationなど)
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
