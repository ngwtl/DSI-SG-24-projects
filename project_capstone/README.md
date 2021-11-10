# Capstone Project - Understanding the cross-section of REIT returns

## Problem Statement

I am a portfolio manager at a real estate fund of funds, and I am intrigued by recent publications in academia on machine learning and asset pricing. According to Campbell and Thompson (2008), small improvements in R2 can lead to large improvements in performance for a mean-variance investor. They show that the proportionate increase in expected excess return earned by an active investor exploiting predictive information is a function of predictive R2 and the Sharpe ratio earned by a passive buy-and-hold investor. Sharpe ratio describes how much excess return an investor's portfolio receives for the volatility he has to endure for holding on to that portfolio, and it is how my performance as a portfolio manager is assessed by my clients. 

My job is to improve my portfolio's Sharpe ratio by finding better ways to assemble my portfolio of real estate stocks to beat a buy-and-hold portfolio such as the FTSE Nareit U.S. Real Estate Index Series. I am encouraged by the recent findings of Gu, Kelly and Xiu (2020) who state that a portfolio that executes market timing on S&P 500 stocks using neural network forecasts enjoys an annualized (out-of-sample) Sharpe ratio of 0.77, as opposed a Sharpe ratio of 0.51 enjoyed by a passive buy-and-hold investor of the S&P 500 index. This is a more than 50% improvement in the risk-return tradeoff. 

## Executive Summary

My dataset starts from 1990 through 2020, comprising of all REITs traded on NYSE, AMEX and NASDAQ. I designated 1990 through 2005 to be the training and validation sets, and 2006 through 2020 for out-of-sample testing. In addition, I made use of 94 stock-level characteristics assembled by Green, Hand and Zhang (2017) as input features for my predictive model.

I managed to achieve an out-of-sample R2 of 0.42 with a 7-layer neural network with L2 regularisation and early stopping. I found that dimension reduction techniques and tree-based models did not work for REITs, unlike Gu, Kelly and Xiu (2020)'s work on the general stock market. In fact, basic regularisation techniques such as lasso and elastic net are the second best-performing prediction models for REITs, while these models are the second worst-performing models for the general stock market. 

The best model from Gu, Kelly and Xiu (2020) is also a neural network, but its performance peaked at 3 layers. They found using deep learning (defined as anything more than 3 layers) results in overfitting to stock market noise. That is not my experience with REITs, as my out-of-sample R2 for a 3-layer network is 0.04, jumping to 0.23 for a 5-layer network, and peaking at 0.42 for a 7-layer network. 



The following table summarizes the performance of all 18 models:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model Name</th>
      <th>Selected Config</th>
      <th>Train (1990-2000)</th>
      <th>Validate (2001-2005)</th>
      <th>Test (2006-2020)</th>
      <th>Test (2006)</th>
      <th>Test (2007)</th>
      <th>Test (2008)</th>
      <th>Test (2009)</th>
      <th>Test (2010)</th>
      <th>Test (2011)</th>
      <th>Test (2012)</th>
      <th>Test (2013)</th>
      <th>Test (2014)</th>
      <th>Test (2015)</th>
      <th>Test (2016)</th>
      <th>Test (2017)</th>
      <th>Test (2018)</th>
      <th>Test (2019)</th>
      <th>Test (2020)</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>naive_reit</td>
      <td>N.A.</td>
      <td>0.00 (10.45)</td>
      <td>0.00 (10.26)</td>
      <td>0.00 (12.13)</td>
      <td>0.00 (7.60)</td>
      <td>0.00 (11.29)</td>
      <td>0.00 (19.81)</td>
      <td>0.00 (25.18)</td>
      <td>0.00 (12.47)</td>
      <td>0.00 (10.08)</td>
      <td>0.00 (9.46)</td>
      <td>0.00 (9.44)</td>
      <td>0.00 (7.04)</td>
      <td>0.00 (7.47)</td>
      <td>0.00 (9.09)</td>
      <td>0.00 (7.08)</td>
      <td>0.00 (8.03)</td>
      <td>0.00 (8.37)</td>
      <td>0.00 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lr_reit</td>
      <td>N.A.</td>
      <td>1.99 (10.3)</td>
      <td>0.69 (10.2)</td>
      <td>-0.03 (12.13)</td>
      <td>4.17 (7.44)</td>
      <td>-3.50 (11.48)</td>
      <td>-0.55 (19.87)</td>
      <td>-0.02 (25.19)</td>
      <td>1.79 (12.36)</td>
      <td>-2.03 (10.19)</td>
      <td>3.18 (9.31)</td>
      <td>1.38 (9.37)</td>
      <td>1.59 (6.99)</td>
      <td>-2.95 (7.58)</td>
      <td>2.22 (8.98)</td>
      <td>-1.00 (7.12)</td>
      <td>-5.43 (8.25)</td>
      <td>3.70 (8.21)</td>
      <td>-0.21 (18.43)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fama-french_reit</td>
      <td>bm, mve0</td>
      <td>0.36 (10.4)</td>
      <td>1.52 (10.2)</td>
      <td>0.17 (12.12)</td>
      <td>3.84 (7.45)</td>
      <td>-4.46 (11.54)</td>
      <td>-1.84 (19.99)</td>
      <td>0.64 (25.10)</td>
      <td>2.03 (12.34)</td>
      <td>-0.68 (10.12)</td>
      <td>3.28 (9.31)</td>
      <td>1.25 (9.38)</td>
      <td>3.36 (6.92)</td>
      <td>-1.87 (7.54)</td>
      <td>2.06 (8.99)</td>
      <td>1.06 (7.05)</td>
      <td>-2.57 (8.14)</td>
      <td>3.50 (8.22)</td>
      <td>-0.05 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>carhart_reit</td>
      <td>bm, mve0, mom12m</td>
      <td>0.52 (10.4)</td>
      <td>1.68 (10.2)</td>
      <td>0.16 (12.12)</td>
      <td>4.37 (7.43)</td>
      <td>-3.46 (11.48)</td>
      <td>-1.33 (19.94)</td>
      <td>0.15 (25.16)</td>
      <td>2.04 (12.34)</td>
      <td>-0.71 (10.12)</td>
      <td>2.65 (9.34)</td>
      <td>1.65 (9.36)</td>
      <td>3.06 (6.93)</td>
      <td>-1.88 (7.54)</td>
      <td>1.71 (9.01)</td>
      <td>0.85 (7.05)</td>
      <td>-2.06 (8.12)</td>
      <td>3.34 (8.23)</td>
      <td>-0.13 (18.43)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>huber_reit</td>
      <td>[3.0, 0.0001]</td>
      <td>1.58 (10.36)</td>
      <td>0.90 (10.21)</td>
      <td>-0.07 (12.13)</td>
      <td>3.50 (7.47)</td>
      <td>-2.04 (11.40)</td>
      <td>-0.24 (19.84)</td>
      <td>-0.24 (25.21)</td>
      <td>0.25 (12.45)</td>
      <td>-0.65 (10.12)</td>
      <td>1.09 (9.41)</td>
      <td>0.70 (9.40)</td>
      <td>2.18 (6.96)</td>
      <td>-2.61 (7.57)</td>
      <td>1.41 (9.02)</td>
      <td>-0.62 (7.11)</td>
      <td>-2.98 (8.15)</td>
      <td>2.99 (8.24)</td>
      <td>-0.31 (18.44)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ridge_reit</td>
      <td>[19920.457084538713]</td>
      <td>1.19 (10.38)</td>
      <td>1.82 (10.16)</td>
      <td>0.18 (12.12)</td>
      <td>4.43 (7.43)</td>
      <td>-3.65 (11.49)</td>
      <td>-1.12 (19.92)</td>
      <td>0.16 (25.16)</td>
      <td>1.98 (12.34)</td>
      <td>-1.29 (10.15)</td>
      <td>3.80 (9.28)</td>
      <td>1.91 (9.35)</td>
      <td>2.74 (6.94)</td>
      <td>-1.82 (7.54)</td>
      <td>2.31 (8.98)</td>
      <td>0.49 (7.07)</td>
      <td>-4.39 (8.21)</td>
      <td>4.05 (8.20)</td>
      <td>-0.13 (18.43)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lasso_reit</td>
      <td>[0.07196856730011514]</td>
      <td>1.28 (10.38)</td>
      <td>1.93 (10.16)</td>
      <td>0.23 (12.11)</td>
      <td>4.37 (7.43)</td>
      <td>-3.54 (11.49)</td>
      <td>-1.09 (19.92)</td>
      <td>0.26 (25.15)</td>
      <td>2.08 (12.34)</td>
      <td>-1.03 (10.14)</td>
      <td>3.42 (9.30)</td>
      <td>1.89 (9.35)</td>
      <td>3.51 (6.92)</td>
      <td>-1.64 (7.54)</td>
      <td>1.70 (9.01)</td>
      <td>0.50 (7.07)</td>
      <td>-4.16 (8.20)</td>
      <td>4.21 (8.19)</td>
      <td>-0.07 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>enet_reit</td>
      <td>[0.9, 0.07880462815669913]</td>
      <td>1.28 (10.38)</td>
      <td>1.93 (10.16)</td>
      <td>0.23 (12.11)</td>
      <td>4.37 (7.43)</td>
      <td>-3.54 (11.49)</td>
      <td>-1.09 (19.92)</td>
      <td>0.26 (25.15)</td>
      <td>2.08 (12.34)</td>
      <td>-1.03 (10.14)</td>
      <td>3.43 (9.30)</td>
      <td>1.90 (9.35)</td>
      <td>3.52 (6.92)</td>
      <td>-1.63 (7.54)</td>
      <td>1.71 (9.01)</td>
      <td>0.52 (7.07)</td>
      <td>-4.17 (8.20)</td>
      <td>4.20 (8.19)</td>
      <td>-0.07 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pcr_reit</td>
      <td>28</td>
      <td>0.90 (10.40)</td>
      <td>1.89 (10.16)</td>
      <td>0.09 (12.12)</td>
      <td>3.79 (7.46)</td>
      <td>-3.68 (11.49)</td>
      <td>-1.31 (19.94)</td>
      <td>0.20 (25.16)</td>
      <td>2.13 (12.33)</td>
      <td>-1.07 (10.14)</td>
      <td>3.27 (9.31)</td>
      <td>1.65 (9.36)</td>
      <td>3.16 (6.93)</td>
      <td>-2.25 (7.56)</td>
      <td>2.12 (8.99)</td>
      <td>0.29 (7.07)</td>
      <td>-4.42 (8.21)</td>
      <td>3.98 (8.20)</td>
      <td>-0.23 (18.44)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pls_reit</td>
      <td>1</td>
      <td>1.03 (10.39)</td>
      <td>1.71 (10.17)</td>
      <td>-0.06 (12.13)</td>
      <td>4.19 (7.44)</td>
      <td>-4.17 (11.52)</td>
      <td>-1.10 (19.92)</td>
      <td>-0.15 (25.20)</td>
      <td>1.75 (12.36)</td>
      <td>-1.84 (10.18)</td>
      <td>3.68 (9.29)</td>
      <td>1.99 (9.34)</td>
      <td>2.32 (6.96)</td>
      <td>-2.06 (7.55)</td>
      <td>2.14 (8.99)</td>
      <td>0.18 (7.08)</td>
      <td>-4.79 (8.22)</td>
      <td>3.94 (8.20)</td>
      <td>-0.42 (18.45)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>rf_reit</td>
      <td>[300, 0.2, 10, 70]</td>
      <td>13.97 (9.69)</td>
      <td>2.31 (10.14)</td>
      <td>-0.75 (12.17)</td>
      <td>4.65 (7.42)</td>
      <td>-11.55 (11.92)</td>
      <td>0.56 (19.76)</td>
      <td>0.03 (25.18)</td>
      <td>3.62 (12.24)</td>
      <td>-2.18 (10.19)</td>
      <td>1.80 (9.38)</td>
      <td>1.17 (9.38)</td>
      <td>2.26 (6.96)</td>
      <td>-2.85 (7.58)</td>
      <td>0.57 (9.06)</td>
      <td>-0.40 (7.10)</td>
      <td>-1.76 (8.10)</td>
      <td>-8.93 (8.73)</td>
      <td>-1.38 (18.54)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>et_reit</td>
      <td>[200, 0.5, 15, 70]</td>
      <td>5.56 (10.15)</td>
      <td>1.63 (10.17)</td>
      <td>-0.04 (12.13)</td>
      <td>4.20 (7.44)</td>
      <td>-4.29 (11.53)</td>
      <td>-0.31 (19.84)</td>
      <td>-0.90 (25.30)</td>
      <td>2.72 (12.30)</td>
      <td>-1.07 (10.14)</td>
      <td>3.04 (9.32)</td>
      <td>1.71 (9.36)</td>
      <td>3.56 (6.91)</td>
      <td>-2.28 (7.56)</td>
      <td>1.87 (9.00)</td>
      <td>-0.46 (7.10)</td>
      <td>-3.92 (8.19)</td>
      <td>2.87 (8.25)</td>
      <td>-0.15 (18.43)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>xg_reit</td>
      <td>[100, 1, 0.01, 0]</td>
      <td>1.00 (10.39)</td>
      <td>1.78 (10.17)</td>
      <td>-0.11 (12.13)</td>
      <td>3.63 (7.46)</td>
      <td>-5.46 (11.59)</td>
      <td>-0.64 (19.88)</td>
      <td>-0.44 (25.24)</td>
      <td>0.98 (12.41)</td>
      <td>-0.64 (10.12)</td>
      <td>2.76 (9.33)</td>
      <td>0.95 (9.39)</td>
      <td>2.64 (6.95)</td>
      <td>-1.44 (7.53)</td>
      <td>1.44 (9.02)</td>
      <td>0.87 (7.05)</td>
      <td>-1.95 (8.11)</td>
      <td>2.80 (8.25)</td>
      <td>-0.15 (18.43)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>nn1-l2_reit</td>
      <td>[&lt;function l2 at 0x7fb45bb4c050&gt;, 3.1622776601683795e-05, 0.001]</td>
      <td>2.84 (10.30)</td>
      <td>1.40 (10.19)</td>
      <td>0.17 (12.12)</td>
      <td>3.04 (7.48)</td>
      <td>-3.14 (11.46)</td>
      <td>-0.98 (19.91)</td>
      <td>0.42 (25.13)</td>
      <td>1.54 (12.37)</td>
      <td>-0.77 (10.12)</td>
      <td>3.98 (9.27)</td>
      <td>2.59 (9.31)</td>
      <td>1.60 (6.98)</td>
      <td>-2.16 (7.56)</td>
      <td>2.41 (8.98)</td>
      <td>-0.05 (7.09)</td>
      <td>-5.10 (8.24)</td>
      <td>2.92 (8.25)</td>
      <td>-0.10 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>nn3-l2_reit</td>
      <td>[&lt;function l2 at 0x7fb45bb4c050&gt;, 0.0001, 0.001]</td>
      <td>3.54 (10.26)</td>
      <td>1.24 (10.19)</td>
      <td>0.04 (12.13)</td>
      <td>2.62 (7.50)</td>
      <td>-3.64 (11.49)</td>
      <td>-0.61 (19.87)</td>
      <td>0.09 (25.17)</td>
      <td>0.67 (12.43)</td>
      <td>-0.18 (10.09)</td>
      <td>1.16 (9.41)</td>
      <td>1.73 (9.36)</td>
      <td>3.07 (6.93)</td>
      <td>-1.32 (7.52)</td>
      <td>1.52 (9.02)</td>
      <td>1.24 (7.04)</td>
      <td>-3.43 (8.17)</td>
      <td>2.10 (8.28)</td>
      <td>-0.02 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>nn5-l2_reit</td>
      <td>[&lt;function l2 at 0x7fb45bb4c050&gt;, 0.0001, 0.01]</td>
      <td>0.84 (10.40)</td>
      <td>1.55 (10.18)</td>
      <td>0.23 (12.11)</td>
      <td>2.50 (7.50)</td>
      <td>-3.51 (11.48)</td>
      <td>-1.69 (19.98)</td>
      <td>0.57 (25.11)</td>
      <td>1.74 (12.36)</td>
      <td>-0.97 (10.13)</td>
      <td>3.50 (9.29)</td>
      <td>1.11 (9.38)</td>
      <td>3.34 (6.92)</td>
      <td>-1.82 (7.54)</td>
      <td>3.42 (8.93)</td>
      <td>1.21 (7.04)</td>
      <td>-2.41 (8.13)</td>
      <td>4.23 (8.19)</td>
      <td>-0.22 (18.44)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>nn7-l2_reit</td>
      <td>[&lt;function l2 at 0x7fbd6a463050&gt;, 3.1622776601683795e-05, 0.01]</td>
      <td>1.47 (10.37)</td>
      <td>1.89 (10.16)</td>
      <td>0.42 (12.10)</td>
      <td>2.09 (7.52)</td>
      <td>-2.02 (11.40)</td>
      <td>-1.01 (19.91)</td>
      <td>0.85 (25.08)</td>
      <td>1.78 (12.36)</td>
      <td>-0.32 (10.10)</td>
      <td>3.32 (9.30)</td>
      <td>1.69 (9.36)</td>
      <td>2.06 (6.97)</td>
      <td>-2.25 (7.56)</td>
      <td>2.09 (8.99)</td>
      <td>1.10 (7.05)</td>
      <td>-3.12 (8.16)</td>
      <td>4.15 (8.19)</td>
      <td>0.00 (18.42)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>nn9-l2_reit</td>
      <td>[&lt;function l2 at 0x7fbd6a463050&gt;, 0.00031622776601683794, 0.01]</td>
      <td>0.66 (10.41)</td>
      <td>1.32 (10.19)</td>
      <td>0.32 (12.11)</td>
      <td>2.45 (7.51)</td>
      <td>-2.08 (11.40)</td>
      <td>-1.07 (19.92)</td>
      <td>0.67 (25.10)</td>
      <td>1.94 (12.35)</td>
      <td>-0.78 (10.12)</td>
      <td>3.03 (9.32)</td>
      <td>1.42 (9.37)</td>
      <td>2.43 (6.96)</td>
      <td>-1.77 (7.54)</td>
      <td>1.63 (9.01)</td>
      <td>0.27 (7.08)</td>
      <td>-2.00 (8.11)</td>
      <td>4.03 (8.20)</td>
      <td>-0.25 (18.44)</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
1
df_r2decrease_reit 

| Model No. | Classifier | CV Score (train) | ROC_AUC (train) | ROC_AUC (test) | Kaggle Score | Runtime (sec) |
|---|---|---|---|---|---|---|
| 1 | LogisticRegression(random_state=42, solver='liblinear') | 0.989000 | 0.994000 | 0.830000 | 0.684000 | 7 |
| 2 | KNeighborsClassifier() | 0.919000 | 1.000000 | 0.7776000 | 0.596000 | 24 |
| 4 | RandomForestClassifier(random_state=42) | 0.979000 | 0.996000 | 0.804000 | 0.643000 | 33 |
| 4 | ExtraTreesClassifier(random_state=42) | 0.989000 | 1.000000 | 0.809000 | 0.710000 | 93 |
| 5 | SVC(max_iter=10000, random_state=42) | 0.973000 | 0.981000 | 0.803000 | 0.598000 | 136 |
| 6 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, <br>subsample=None, tree_method=None, validate_parameters=None, verbosity=None) | 0.992000 | 1.000000 | 0.831000 | 0.697000 | 133 |
| 7 | LogisticRegression(random_state=42) | 0.989000 | 0.994000 | 0.828000 | 0.684000 | 176 |
| 8 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, <br>subsample=None, tree_method=None, validate_parameters=None, verbosity=None) | 0.991000 | 0.999000 | 0.830000 | 0.658000 | 104 |
| 9 | LogisticRegression(random_state=42, solver='liblinear') | 0.989000 | 0.994000 | 0.826000 | 0.693000 | 10 |
| 10 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, subsample=None, <br>tree_method=None, validate_parameters=None, verbosity=None) | 0.990000 | 1.000000 | 0.797000 | 0.673000 | 182 |
| 11 | LogisticRegression(random_state=42, solver='liblinear') | 0.988000 | 0.993000 | 0.832000 | 0.717000 | 6 |
| 12 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, <br>subsample=None, tree_method=None, validate_parameters=None, verbosity=None) | 0.990000 | 0.997000 | 0.826000 | 0.683000 | 141 |
| 13 | LogisticRegression(random_state=42, solver='liblinear') | 0.987000 | 0.992000 | 0.826000 | 0.734000 | 4 |
| 14 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, subsample=None, <br>tree_method=None, validate_parameters=None, verbosity=None) | 0.992000 | 0.999000 | 0.825000 | 0.715000 | 114 |
| 15 | LogisticRegression(random_state=42, solver='liblinear') | 0.989000 | 0.993000 | 0.845000 | 0.708000 | 5 |
| 16 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, subsample=None, <br>tree_method=None, validate_parameters=None, verbosity=None) | 0.990000 | 0.997000 | 0.829000 | 0.690000 | 118 |
| 17 | LogisticRegression(random_state=42, solver='liblinear') | 0.990000 | 0.994000 | 0.849000 | 0.725000 | 9 |
| 18 | XGBClassifier(base_score=None, booster=None, colsample_bylevel=None, <br>colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=None, <br>importance_type='gain', interaction_constraints=None, learning_rate=None, <br>max_delta_step=None, max_depth=None, min_child_weight=None, missing=nan, <br>monotone_constraints=None, n_estimators=100, n_jobs=None, num_parallel_tree=None, <br>random_state=42, reg_alpha=None, reg_lambda=None, scale_pos_weight=None, subsample=None, <br>tree_method=None, validate_parameters=None, verbosity=None) | 0.991000 | 0.999000 | 0.843000 | 0.704000 | 129 |
|  |  |  |  |  |  |  |


## Data Dictionary

| Feature | Type | Dataset | Description |
|---|---|---|---|
| Id | Integer | Test | The id of the record |
| Date | Object | Train/Test/Weather | Date that the WNV test is performed for Train/TestDate of weather measurement for Weather |
| Address | Object | Train/Test | Approximate address of the location of trap. This is used to send to the GeoCoder. |
| Species | Object | Train/Test | The species of mosquitos |
| Block | Integer | Train/Test | Block number of address |
| Street | Object | Train/Test | Street name |
| Trap | Object | Train/Test | Id of the trap |
| AddressNumberAndStreet | Object | Train/Test | Approximate address returned from GeoCoder |
| Latitude | Float | Train/Test/Spray | Latitude returned from GeoCoder for Train/Test <br>Latitude of the spray |
| Longitude | Float | Train/Test/Spray | Longitude returned from GeoCoder for Train/Test<br>Longitude of the spray |
| AddressAccuracy | Integer | Train/Test | Accuracy returned from GeoCoder |
| NumMosquitos | Integer | Train | Number of mosquitoes caught in this trap |
| WnvPresent | Integer | Train | Whether West Nile Virus was present in these mosquitos. <br>1 means WNV is present, and 0 means not present. |
| Station | Integer | Weather | 1; automated station without a precipitation descriminator. <br>2; automated station with precipitation descriminator. |
| Tmax | Integer | Weather | Maximum Temperature |
| Tmin | Integer | Weather | Minimum Temperature |
| Tavg | Integer | Weather | Average Temperature |
| Depart | Integer | Weather | Departure from normal |
| Dewpoint | Integer | Weather | Average dew point |
| Wetbulb | Integer | Weather | Average wet bulb |
| Heat | Integer | Weather | Heating (Season begins with July) |
| Cool | Integer | Weather | Cooling (Season begins with January) |
| Sunrise | Object | Weather | Sunrise Time (Calculated, not observed) |
| Sunset | Object | Weather | Sunset Time (Calculated, not observed) |
| CodeSum | Object | Weather | Significant Weather Types<br><br>+FC TORNADO/WATERSPOUT<br> FC  FUNNEL CLOUD<br> TS  THUNDERSTORM<br> GR  HAIL<br>RA RAIN<br>DZ DRIZZLE<br>SN SNOW<br>SG SNOW GRAINS<br>GS SMALL HAIL &/OR SNOW PELLETS<br>PL ICE PELLETS<br>IC ICE CRYSTALS<br>FG+ HEAVY FOG (FG & LE.25 MILES VISIBILITY) FG FOG<br>BR MIST<br>UP UNKNOWN PRECIPITATION<br>HZ HAZE<br>FU SMOKE<br>VA VOLCANIC ASH<br>DU WIDESPREAD DUST<br>DS DUSTSTORM<br>PO SAND/DUST WHIRLS<br>SA SAND<br>SS SANDSTORM<br>PY SPRAY<br>SQ SQUALL<br>DR LOW DRIFTING<br>SH SHOWER<br>FZ FREEZING<br>MI SHALLOW<br>PR PARTIAL<br>BC PATCHES<br>BL BLOWING<br>VC VICINITY<br>- LIGHT + HEAVY<br>"NO SIGN" MODERATE |
| Depth | Integer | Weather | Depth of snow/ice in inches SNOW/ICE (ON GROUND)(1200 UTC)T = TRACEM = MISSING DATA |
| Water1 | Integer | Weather | WATER EQUIVALENT (1800 UTC) M = MISSING DATA |
| SnowFall | Float | Weather | SNOWFALL (INCHES AND TENTHS)(2400 LST)*<br>T = TRACEM = MISSING DATA |
| PrecipTotal | Float | Weather | WATER EQUIVALENT(INCHES & HUNDREDTHS(2400 LST) RAINFALL & MELTED SNOW<br>M = MISSING DATAT = TRACE |
| StnPressure | Float | Weather | Pressure in inches |
| SeaLevel | Float | Weather | Average Sea Level Pressure |
| ResultSpeed | Float | Weather | Resultant Wind Speed |
| ResultDir | Integer | Weather | Resultant Wind Direction (Whole Degree) |
| AvgSpeed | Float | Weather | Wing Average Speed |


