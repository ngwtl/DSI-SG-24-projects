# Capstone Project - Understanding and predicting the cross-section of REIT returns

## Problem Statement

I am a portfolio manager at a real estate fund of funds, and I am intrigued by recent publications in academia on machine learning and asset pricing. According to Campbell and Thompson (2008), small improvements in R2 can lead to large improvements in performance for a mean-variance investor. They show that the proportionate increase in expected excess return earned by an active investor exploiting predictive information is a function of predictive R2 and the Sharpe ratio earned by a passive buy-and-hold investor. Sharpe ratio describes how much excess return an investor's portfolio receives for the volatility he has to endure for holding on to that portfolio, and it is how my performance as a portfolio manager is assessed by my clients. 

My job is to improve my portfolio's Sharpe ratio by finding better ways to assemble my portfolio of real estate stocks to beat a buy-and-hold portfolio such as the FTSE Nareit U.S. Real Estate Index Series. I am encouraged by the recent findings of Gu, Kelly and Xiu (2020) who state that a portfolio that executes market timing on S&P 500 stocks using neural network forecasts enjoys an annualized (out-of-sample) Sharpe ratio of 0.77, as opposed a Sharpe ratio of 0.51 enjoyed by a passive buy-and-hold investor of the S&P 500 index. This is a more than 50% improvement in the risk-return tradeoff. 

## Executive Summary

My dataset starts from 1990 through 2020, comprising of all REITs traded on NYSE, AMEX and NASDAQ. I designated 1990 through 2005 to be the training and validation sets, and 2006 through 2020 for out-of-sample testing. In addition, I made use of 94 stock-level characteristics assembled by Green, Hand and Zhang (2017) as input features for my predictive model.

I managed to achieve an out-of-sample R2 of 0.42 with a 7-layer neural network with L2 regularisation and early stopping. I found that dimension reduction techniques and tree-based models did not work for REITs, unlike Gu, Kelly and Xiu (2020)'s work on the general stock market. In fact, basic regularisation techniques such as lasso and elastic net are the second best-performing prediction models for REITs, while these models are the second worst-performing models for the general stock market. 

The best model from Gu, Kelly and Xiu (2020) is also a neural network, but its performance peaked at 3 layers. They found using deep learning (defined as anything more than 3 layers) results in overfitting to stock market noise. That is not my experience with REITs, as my out-of-sample R2 for a 3-layer network is 0.04, jumping to 0.23 for a 5-layer network, and peaking at 0.42 for a 7-layer network. 

The following table summarizes the performance of all 18 models. The numbers without parentheses are R2s, while those in parentheses are RMSE.

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


## Data Dictionary

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature acronym</th>
      <th>Original author(s)</th>
      <th>Date, journal</th>
      <th>Description of feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>absacc</td>
      <td>Bandyopadhyay, Huang, and Wirjanto</td>
      <td>2010, WP</td>
      <td>Absolute value ofacc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>acc</td>
      <td>Sloan</td>
      <td>1996, TAR</td>
      <td>Annual income before extraordinary items (ib) minus operating cash flows (oancf) divided by average total assets (at); ifoancfis missing then set to change inact- change inche- change inlct|$+$|change indlc|$+$|change intxp-dp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aeavol</td>
      <td>Lerman, Livnat, and Mendenhall</td>
      <td>2008, WP</td>
      <td>Average daily trading volume (vol) for 3 days around earnings announcement minus average daily volume for 1-month ending 2 weeks before earnings announcement divided by 1-month average daily volume. Earnings announcement day from Compustat quarterly (rdq)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>age</td>
      <td>Jiang, Lee, and Zhang</td>
      <td>2005, RAS</td>
      <td>Number of years since first Compustat coverage</td>
    </tr>
    <tr>
      <th>4</th>
      <td>agr</td>
      <td>Cooper, Gulen, and Schill</td>
      <td>2008, JF</td>
      <td>Annual percent change in total assets (at)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>baspread</td>
      <td>Amihud and Mendelson</td>
      <td>1989, JF</td>
      <td>Monthly average of daily bid-ask spread divided by average of daily spread</td>
    </tr>
    <tr>
      <th>6</th>
      <td>beta</td>
      <td>Fama and MacBeth</td>
      <td>1973, JPE</td>
      <td>Estimated market beta from weekly returns and equal weighted market returns for 3 years ending month|$t$|-1 with at least 52 weeks of returns</td>
    </tr>
    <tr>
      <th>7</th>
      <td>betasq</td>
      <td>Fama and MacBeth</td>
      <td>1973, JPE</td>
      <td>Market beta squared</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bm</td>
      <td>Rosenberg, Reid, and Lanstein</td>
      <td>1985, JPM</td>
      <td>Book value of equity (ceq) divided by end of fiscal year-end market capitalization</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bm_ia</td>
      <td>Asness, Porter, and Stevens</td>
      <td>2000, WP</td>
      <td>Industry adjusted book-to-market ratio</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cash</td>
      <td>Palazzo</td>
      <td>2012, JFE</td>
      <td>Cash and cash equivalents divided by average total assets</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cashdebt</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Earnings before depreciation and extraordinary items (ib|$+$|dp) divided by avg. total liabilities (lt)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cashpr</td>
      <td>Chandrashekar and Rao</td>
      <td>2009, WP</td>
      <td>Fiscal year-end market capitalization plus long-term debt (dltt) minus total assets (at) divided by cash and equivalents (che)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cfp</td>
      <td>Desai, Rajgopal, and Venkatachalam</td>
      <td>2004, TAR</td>
      <td>Operating cash flows divided by fiscal-year-end market capitalization</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cfp_ia</td>
      <td>Asness, Porter and Stevens</td>
      <td>2000, WP</td>
      <td>Industry adjustedcfp</td>
    </tr>
    <tr>
      <th>15</th>
      <td>chatoia</td>
      <td>Soliman</td>
      <td>2008, TAR</td>
      <td>2-digit SIC - fiscal-year mean-adjusted change in sales (sale) divided by average total assets (at)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>chcsho</td>
      <td>Pontiff and Woodgate</td>
      <td>2008, JF</td>
      <td>Annual percent change in shares outstanding (csho)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>chempia</td>
      <td>Asness, Porter, and Stevens</td>
      <td>1994, WP</td>
      <td>Industry-adjusted change in number of employees</td>
    </tr>
    <tr>
      <th>18</th>
      <td>chfeps</td>
      <td>Hawkins, Chamberlin, and Daniel</td>
      <td>1984, FAJ</td>
      <td>Mean analyst forecast in month prior to fiscal period end date from I/B/E/S summary file minus same mean forecast for prior fiscal period using annual earnings forecasts</td>
    </tr>
    <tr>
      <th>19</th>
      <td>chinv</td>
      <td>Thomas and Zhang</td>
      <td>2002, RAS</td>
      <td>Change in inventory (inv) scaled by average total assets (at)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>chmom</td>
      <td>Gettleman and Marks</td>
      <td>2006, WP</td>
      <td>Cumulative returns from months|$t$|-6 to|$t$|-1 minus months|$t$|-12 to|$t$|-7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>chnanalyst</td>
      <td>Scherbina</td>
      <td>2008 RF</td>
      <td>Change innanalystfrom month|$t$|-3 to month|$t$|</td>
    </tr>
    <tr>
      <th>22</th>
      <td>chpmia</td>
      <td>Soliman</td>
      <td>2008, TAR</td>
      <td>2-digit SIC - fiscal-year mean adjusted change in income before extraordinary items (ib) divided by sales (sale)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>chtx</td>
      <td>Thomas and Zhang</td>
      <td>2011, JAR</td>
      <td>Percent change in total taxes (txtq) from quarter|$ t$|-4 to|$t$|</td>
    </tr>
    <tr>
      <th>24</th>
      <td>cinvest</td>
      <td>Titman, Wei, and Xie</td>
      <td>2004, JFQA</td>
      <td>Change over one quarter in net PP&amp;E (ppentq) divided by sales (saleq) - average of this variable for prior 3 quarters; ifsaleq|$=$|0, then scale by 0.01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>convind</td>
      <td>Valta</td>
      <td>2016, JFQA</td>
      <td>An indicator equal to 1 if company has convertible debt obligations</td>
    </tr>
    <tr>
      <th>26</th>
      <td>currat</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Current assets / current liabilities</td>
    </tr>
    <tr>
      <th>27</th>
      <td>depr</td>
      <td>Holthausen and Larcker</td>
      <td>1992, JAE</td>
      <td>Depreciation divided by PP&amp;E</td>
    </tr>
    <tr>
      <th>28</th>
      <td>disp</td>
      <td>Diether, Malloy, and Scherbina</td>
      <td>2002, JF</td>
      <td>Standard deviation of analyst forecasts in month prior to fiscal period end date divided by the absolute value of the mean forecast; ifmeanest|$=$|0, then scalar set to 1. Forecast data from I/B/E/S summary files</td>
    </tr>
    <tr>
      <th>29</th>
      <td>divi</td>
      <td>Michaely, Thaler, and Womack</td>
      <td>1995, JF</td>
      <td>An indicator variable equal to 1 if company pays dividends but did not in prior year</td>
    </tr>
    <tr>
      <th>30</th>
      <td>divo</td>
      <td>Michaely, Thaler, and Womack</td>
      <td>1995, JF</td>
      <td>An indicator variable equal to 1 if company does not pay dividend but did in prior year</td>
    </tr>
    <tr>
      <th>31</th>
      <td>dolvol</td>
      <td>Chordia, Subrahmanyam, and Anshuman</td>
      <td>2001, JFE</td>
      <td>Natural log of trading volume times price per share from month|$t$|-2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>dy</td>
      <td>Litzenberger and Ramaswamy</td>
      <td>1982, JF</td>
      <td>Total dividends (dvt) divided by market capitalization at fiscal year-end</td>
    </tr>
    <tr>
      <th>33</th>
      <td>ear</td>
      <td>Kishore et al.</td>
      <td>2008, WP</td>
      <td>Sum of daily returns in three days around earnings announcement. Earnings announcement from Compustat quarterly file (rdq)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>egr</td>
      <td>Richardson et al.</td>
      <td>2005, JAE</td>
      <td>Annual percent change in book value of equity (ceq)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>ep</td>
      <td>Basu</td>
      <td>1977, JF</td>
      <td>Annual income before extraordinary items (ib) divided by end of fiscal year market cap</td>
    </tr>
    <tr>
      <th>36</th>
      <td>fgr5yr</td>
      <td>Bauman and Dowen</td>
      <td>1988, FAJ</td>
      <td>Most recently available analyst forecasted 5-year growth</td>
    </tr>
    <tr>
      <th>37</th>
      <td>gma</td>
      <td>Novy-Marx</td>
      <td>2013, JFE</td>
      <td>Revenues (revt) minus cost of goods sold (cogs) divided by lagged total assets (at)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>grCAPX</td>
      <td>Anderson and Garcia-Feijoo</td>
      <td>2006, JF</td>
      <td>Percent change in capital expenditures from year|$ t$|-2 to year|$t$|</td>
    </tr>
    <tr>
      <th>39</th>
      <td>grltnoa</td>
      <td>Fairfield, Whisenant, and Yohn</td>
      <td>2003, TAR</td>
      <td>Growth in long-term net operating assets</td>
    </tr>
    <tr>
      <th>40</th>
      <td>herf</td>
      <td>Hou and Robinson</td>
      <td>2006, JF</td>
      <td>2-digit SIC - fiscal-year sales concentration (sum of squared percent of sales in industry for each company).</td>
    </tr>
    <tr>
      <th>41</th>
      <td>hire</td>
      <td>Bazdresch, Belo, and Lin</td>
      <td>2014, JPE</td>
      <td>Percent change in number of employees (emp)</td>
    </tr>
    <tr>
      <th>42</th>
      <td>idiovol</td>
      <td>Ali, Hwang, and Trombley</td>
      <td>2003, JFE</td>
      <td>Standard deviation of residuals of weekly returns on weekly equal weighted market returns for 3 years prior to month end</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ill</td>
      <td>Amihud</td>
      <td>2002, JFM</td>
      <td>Average of daily (absolute return / dollar volume).</td>
    </tr>
    <tr>
      <th>44</th>
      <td>indmom</td>
      <td>Moskowitz and Grinblatt</td>
      <td>1999, JF</td>
      <td>Equal weighted average industry 12-month returns</td>
    </tr>
    <tr>
      <th>45</th>
      <td>invest</td>
      <td>Chen and Zhang</td>
      <td>2010, JF</td>
      <td>Annual change in gross property, plant, and equipment (ppegt)|$+$|annual change in inventories (invt) all scaled by lagged total assets (at)</td>
    </tr>
    <tr>
      <th>46</th>
      <td>IPO</td>
      <td>Loughran and Ritter</td>
      <td>1995, JF</td>
      <td>An indicator variable equal to 1 if first year available on CRSP monthly stock file</td>
    </tr>
    <tr>
      <th>47</th>
      <td>lev</td>
      <td>Bhandari</td>
      <td>1988, JF</td>
      <td>Total liabilities (lt) divided by fiscal year-end market capitalization</td>
    </tr>
    <tr>
      <th>48</th>
      <td>lgr</td>
      <td>Richardson et al.</td>
      <td>2005, JAE</td>
      <td>Annual percent change in total liabilities (lt)</td>
    </tr>
    <tr>
      <th>49</th>
      <td>maxret</td>
      <td>Bali, Cakici, and Whitelaw</td>
      <td>2011, JFE</td>
      <td>Maximum daily return from returns during calendar month|$ t$|-1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>mom12m</td>
      <td>Jegadeesh</td>
      <td>1990, JF</td>
      <td>11-month cumulative returns ending one month before month end</td>
    </tr>
    <tr>
      <th>51</th>
      <td>mom1m</td>
      <td>Jegadeesh and Titman</td>
      <td>1993, JF</td>
      <td>1-month cumulative return</td>
    </tr>
    <tr>
      <th>52</th>
      <td>mom36m</td>
      <td>Jegadeesh and Titman</td>
      <td>1993, JF</td>
      <td>Cumulative returns from months|$ t$|-36 to|$t$|-13</td>
    </tr>
    <tr>
      <th>53</th>
      <td>mom6m</td>
      <td>Jegadeesh and Titman</td>
      <td>1993, JF</td>
      <td>5-month cumulative returns ending one month before month end</td>
    </tr>
    <tr>
      <th>54</th>
      <td>ms</td>
      <td>Mohanram</td>
      <td>2005, RAS</td>
      <td>Sum of 8 indicator variables for fundamental performance</td>
    </tr>
    <tr>
      <th>55</th>
      <td>mve</td>
      <td>Banz</td>
      <td>1981, JFE</td>
      <td>Natural log of market capitalization at end of month|$t$|-1</td>
    </tr>
    <tr>
      <th>56</th>
      <td>mve_ia</td>
      <td>Asness, Porter, and Stevens</td>
      <td>2000, WP</td>
      <td>2-digit SIC industry-adjusted fiscal year-end market capitalization</td>
    </tr>
    <tr>
      <th>57</th>
      <td>nanalyst</td>
      <td>Elgers, Lo, and Pfeiffer</td>
      <td>2001, TAR</td>
      <td>Number of analyst forecasts from most recently available I/B/E/S summary files in month prior to month of portfolio formation.nanalystset to zero if not covered in I/B/E/S summary file</td>
    </tr>
    <tr>
      <th>58</th>
      <td>nincr</td>
      <td>Barth, Elliott, and Finn</td>
      <td>1999, JAR</td>
      <td>Number of consecutive quarters (up to eight quarters) with an increase in earnings (ibq) over same quarter in the prior year</td>
    </tr>
    <tr>
      <th>59</th>
      <td>operprof</td>
      <td>Fama and French</td>
      <td>2015, JFE</td>
      <td>Revenue minus cost of goods sold - SG&amp;A expense - interest expense divided by lagged common shareholders’ equity</td>
    </tr>
    <tr>
      <th>60</th>
      <td>orgcap</td>
      <td>Eisfeldt and Papanikolaou</td>
      <td>2013, JF</td>
      <td>Capitalized SG&amp;A expenses</td>
    </tr>
    <tr>
      <th>61</th>
      <td>pchcapx_ia</td>
      <td>Abarbanell and Bushee</td>
      <td>1998, TAR</td>
      <td>2-digit SIC - fiscal-year mean-adjusted percent change in capital expenditures (capx)</td>
    </tr>
    <tr>
      <th>62</th>
      <td>pchcurrat</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Percent change incurrat.</td>
    </tr>
    <tr>
      <th>63</th>
      <td>pchdepr</td>
      <td>Holthausen and Larcker</td>
      <td>1992, JAE</td>
      <td>Percent change indepr</td>
    </tr>
    <tr>
      <th>64</th>
      <td>pchgm_pchsale</td>
      <td>Abarbanell and Bushee</td>
      <td>1998, TAR</td>
      <td>Percent change in gross margin (sale-cogs) minus percent change in sales (sale)</td>
    </tr>
    <tr>
      <th>65</th>
      <td>pchquick</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Percent change inquick</td>
    </tr>
    <tr>
      <th>66</th>
      <td>pchsale_pchinvt</td>
      <td>Abarbanell and Bushee</td>
      <td>1998, TAR</td>
      <td>Annual percent change in sales (sale) minus annual percent change in inventory (invt).</td>
    </tr>
    <tr>
      <th>67</th>
      <td>pchsale_pchrect</td>
      <td>Abarbanell and Bushee</td>
      <td>1998, TAR</td>
      <td>Annual percent change in sales (sale) minus annual percent change in receivables (rect)</td>
    </tr>
    <tr>
      <th>68</th>
      <td>pchsale_pchxsga</td>
      <td>Abarbanell and Bushee</td>
      <td>1998, TAR</td>
      <td>Annual percent change in sales (sale) minus annual percent change in SG&amp;A (xsga)</td>
    </tr>
    <tr>
      <th>69</th>
      <td>pchsaleinv</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Percent change insaleinv</td>
    </tr>
    <tr>
      <th>70</th>
      <td>pctacc</td>
      <td>Hafzalla, Lundholm, and Van Winkle</td>
      <td>2011, TAR</td>
      <td>Same asaccexcept that the numerator is divided by the absolute value ofib; ifib|$=$|0 thenibset to 0.01 for denominator</td>
    </tr>
    <tr>
      <th>71</th>
      <td>pricedelay</td>
      <td>Hou &amp; Moskowitz</td>
      <td>2005, RFS</td>
      <td>The proportion of variation in weekly returns for 36 months ending in month|$ t$|explained by 4 lags of weekly market returns incremental to contemporaneous market return</td>
    </tr>
    <tr>
      <th>72</th>
      <td>ps</td>
      <td>Piotroski</td>
      <td>2000, JAR</td>
      <td>Sum of 9 indicator variables to form fundamental health score</td>
    </tr>
    <tr>
      <th>73</th>
      <td>quick</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>(current assets - inventory) / current liabilities</td>
    </tr>
    <tr>
      <th>74</th>
      <td>rd</td>
      <td>Eberhart, Maxwell, and Siddique</td>
      <td>2004, JF</td>
      <td>An indicator variable equal to 1 if R&amp;D expense as a percentage of total assets has an increase greater than 5%.</td>
    </tr>
    <tr>
      <th>75</th>
      <td>rd_mve</td>
      <td>Guo, Lev, and Shi</td>
      <td>2006, JBFA</td>
      <td>R&amp;D expense divided by end-of-fiscal-year market capitalization</td>
    </tr>
    <tr>
      <th>76</th>
      <td>rd_sale</td>
      <td>Guo, Lev, and Shi</td>
      <td>2006, JBFA</td>
      <td>R&amp;D expense divided by sales (xrd/sale)</td>
    </tr>
    <tr>
      <th>77</th>
      <td>realestate</td>
      <td>Tuzel</td>
      <td>2010, RFS</td>
      <td>Buildings and capitalized leases divided by gross PP&amp;E</td>
    </tr>
    <tr>
      <th>78</th>
      <td>retvol</td>
      <td>Ang et al.</td>
      <td>2006, JF</td>
      <td>Standard deviation of daily returns from month|$t$|-1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>roaq</td>
      <td>Balakrishnan, Bartov, and Faurel</td>
      <td>2010, JAE</td>
      <td>Income before extraordinary items (ibq) divided by one quarter lagged total assets (atq)</td>
    </tr>
    <tr>
      <th>80</th>
      <td>roavol</td>
      <td>Francis et al.</td>
      <td>2004, TAR</td>
      <td>Standard deviation for 16 quarters of income before extraordinary items (ibq) divided by average total assets (atq)</td>
    </tr>
    <tr>
      <th>81</th>
      <td>roeq</td>
      <td>Hou, Xue, and Zhang</td>
      <td>2015 RFS</td>
      <td>Earnings before extraordinary items divided by lagged common shareholders’ equity</td>
    </tr>
    <tr>
      <th>82</th>
      <td>roic</td>
      <td>Brown and Rowe</td>
      <td>2007, WP</td>
      <td>Annual earnings before interest and taxes (ebit) minus nonoperating income (nopi) divided by non-cash enterprise value (ceq|$+$|lt-che)</td>
    </tr>
    <tr>
      <th>83</th>
      <td>rsup</td>
      <td>Kama</td>
      <td>2009, JBFA</td>
      <td>Sales from quarter t minus sales from quarter|$t$|-4 (saleq) divided by fiscal-quarter-end market capitalization (cshoq*prccq)</td>
    </tr>
    <tr>
      <th>84</th>
      <td>salecash</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Annual sales divided by cash and cash equivalents</td>
    </tr>
    <tr>
      <th>85</th>
      <td>saleinv</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Annual sales divided by total inventory</td>
    </tr>
    <tr>
      <th>86</th>
      <td>salerec</td>
      <td>Ou and Penman</td>
      <td>1989, JAE</td>
      <td>Annual sales divided by accounts receivable</td>
    </tr>
    <tr>
      <th>87</th>
      <td>secured</td>
      <td>Valta</td>
      <td>2016, JFQA</td>
      <td>Total liability scaled secured debt</td>
    </tr>
    <tr>
      <th>88</th>
      <td>securedind</td>
      <td>Valta</td>
      <td>2016, JFQA</td>
      <td>An indicator equal to 1 if company has secured debt obligations</td>
    </tr>
    <tr>
      <th>89</th>
      <td>sfe</td>
      <td>Elgers, Lo, and Pfeiffer</td>
      <td>2001, TAR</td>
      <td>Analysts mean annual earnings forecast for nearest upcoming fiscal year from most recent month available prior to month of portfolio formation from I/B/E/S summary files scaled by price per share at fiscal quarter end</td>
    </tr>
    <tr>
      <th>90</th>
      <td>sgr</td>
      <td>Lakonishok, Shleifer, and Vishny</td>
      <td>1994, JF</td>
      <td>Annual percent change in sales (sale)</td>
    </tr>
    <tr>
      <th>91</th>
      <td>sin</td>
      <td>Hong &amp; Kacperczyk</td>
      <td>2009, JFE</td>
      <td>An indicator variable equal to 1 if a company’s primary industry classification is in smoke or tobacco, beer or alcohol, or gaming</td>
    </tr>
    <tr>
      <th>92</th>
      <td>SP</td>
      <td>Barbee, Mukherji, and Raines</td>
      <td>1996, FAJ</td>
      <td>Annual revenue (sale) divided by fiscal year-end market capitalization</td>
    </tr>
    <tr>
      <th>93</th>
      <td>std_dolvol</td>
      <td>Chordia, Subrahmanyam, and Anshuman</td>
      <td>2001, JFE</td>
      <td>Monthly standard deviation of daily dollar trading volume</td>
    </tr>
    <tr>
      <th>94</th>
      <td>std_turn</td>
      <td>Chordia, Subrahmanyam, and Anshuman</td>
      <td>2001, JFE</td>
      <td>Monthly standard deviation of daily share turnover</td>
    </tr>
    <tr>
      <th>95</th>
      <td>stdacc</td>
      <td>Bandyopadhyay, Huang, and Wirjanto</td>
      <td>2010, WP</td>
      <td>Standard deviation for 16 quarters of accruals (accmeasured with quarterly Compustat) scaled by sales; ifsaleq|$=$|0, then scale by 0.01</td>
    </tr>
    <tr>
      <th>96</th>
      <td>stdcf</td>
      <td>Huang</td>
      <td>2009, JEF</td>
      <td>Standard deviation for 16 quarters of cash flows divided by sales (saleq); ifsaleq|$=$|0, then scale by 0.01. Cash flows defined asibqminus quarterly accruals</td>
    </tr>
    <tr>
      <th>97</th>
      <td>sue</td>
      <td>Rendelman, Jones, and Latane</td>
      <td>1982, JFE</td>
      <td>Unexpected quarterly earnings divided by fiscal-quarter-end market cap. Unexpected earnings is I/B/E/S actual earnings minus median forecasted earnings if available, else it is the seasonally differenced quarterly earnings before extraordinary items from Compustat quarterly file</td>
    </tr>
    <tr>
      <th>98</th>
      <td>tang</td>
      <td>Almeida and Campello</td>
      <td>2007, RFS</td>
      <td>Cash holdings|$+$|0.715|$\times$|receivables|$+$|0.547|$\times$|inventory|$+$|0.535|$\times$|PPE/ totl assets</td>
    </tr>
    <tr>
      <th>99</th>
      <td>tb</td>
      <td>Lev and Nissim</td>
      <td>2004, TAR</td>
      <td>Tax income, calculated from current tax expense divided by maximum federal tax rate, divided by income before extraordinary items</td>
    </tr>
    <tr>
      <th>100</th>
      <td>turn</td>
      <td>Datar, Naik, and Radcliffe</td>
      <td>1998, JFM</td>
      <td>Average monthly trading volume for most recent 3 months scaled by number of shares outstanding in current month</td>
    </tr>
    <tr>
      <th>101</th>
      <td>zerotrade</td>
      <td>Liu</td>
      <td>2006, JFE</td>
      <td>Turnover weighted number of zero trading days for most recent 1 month</td>
    </tr>
  </tbody>
</table>
