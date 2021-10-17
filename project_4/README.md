# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4 - West Nile Virus Prediction

## Problem Statement

Our team, as part of the Chicago Department of Public Health (CDPH), has been tasked with predicting the next outbreak of the West Nile Virus (WNV) in Chicago. This information would be useful for the City of Chicago and the CDPH to identify areas of high risk of outbreak in advance and to allocate resources accordingly. We will also perform a cost-benefit analysis of the current vector control program and make recommendation on the next steps going forward.

Due to the recent epidemic of West Nile Virus in the Windy City, we've had the Department of Public Health set up a surveillance and control system. We're hoping it will let us learn something from the mosquito population as we collect data over time. Pesticides are a necessary evil in the fight for public health and safety, not to mention expensive! We need to derive an effective plan to deploy pesticides throughout the city, and that is **exactly** where you come in!

## Executive Summary

For this project, datasets containing information such as mosquitos species caught, number of mosquitoes caught, presence of WNV in the mosquitos and coordinate location of traps were used. Information on the date and location of spraying and various weather parameters were also utilized. The following summarises the analysis process:

1. Impute or remove missing or duplicated data from the train, test, weather and spray datasets

2. Merge the train and spray datasets and perform EDA on the effect of spraying on presence of WNV and number of mosquitos

3. Merge the train and weather datasets and perform feature engineering and selection by evaluating the ROC_AUC score of models against the selected features iteratively 

4. Tune the hyperparameters for 18 models by gridsearching and choose the best one based on ROC_AUC score on the train set, the CV ROC_AUC score on the train set , the ROC_AUC score on the test set and finally the Kaggle score

The best model selected was the XGBoost Classifier (Model 18) as even though it has a lower ROC_AUC than the Logistic Regression (Model 17), it has a higher sensitivity which is important because Chicago city officials might make expensive decisions on mosquito-eradication methods (spray in areas we predict to be positive, and don't spray in areas we predict to be negative). Therefore it is also important to have as high true positive as possible, and as low false negative as possible. This points to using sensitivity as a secondary metric for us to use as assessment since ROC_AUC is quite close between Models 17 and 18.

The following table summarizes the performance of all 18 models:

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
