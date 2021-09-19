# Project 2 - Ames Housing Data and Kaggle Challenge
## by Ethan Leow


### Problem Statement
I work for a real estate consultancy in Ames, IA. My boss has tasked me to come up with a good predictive model for house prices in my city. The main purpose is to help our clients, i.e. buyers of homes, shortlist undervalued houses and understand features of a property that are important drivers of sale prices. To do so, I need to come up with a predictive model that can generate an expected value for a property given all its attributes, and the client can compare it with the asking price of the seller.

### Data Dictionary
The original data dictionary is from Dean De Cock of Truman State University):
[data dictionary](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt).
The following is my final data dictionary after data cleaning (fixing and imputing null values), basic feature engineering (conversion to dummies, assigning ordinal values to descriptions) and feature selection (screening out variables with high correlation to one another, and low correlation to sale prices). From the original 82 features, I selected 29 to be used for modelling and one final round of feature selection.

The cleaned data sets are stored in "../datasets/df_train.csv" and "../datasets/df_test.csv".

| No. | Feature | DType |  Description | Remarks |
|-----|---------|------|-------------|---------|
| 1 | saleprice | integer | Sale price $ | \$12,789 to  \$611,657 |
| 2 | street | integer | Type of road access to property (Paved or Gravel) | Pave = 1, Gravel = 0 | 
| 3 | neighborhood | integer | Physical locations within Ames city limits | Dummified 28 neighborhoods, dropped "Old Town" to become the baseline |
| 4 | condition_1 | integer | Proximity to various conditions ("Norm" = Normal, "FeederArt" = Adjacent to arterial street or feeder street, "Pos" = Adjacent or near positive off-site feature, "Rail" = Adjacent or near North-South Railroad or East-West Railroad) | Dummified the 4 conditions, dropped "Norm" to become the baseline |
| 5 | overall_qual | integer | Rates the overall material and finish of the house | 1 (very poor) to 10 (very excellent) |
| 6 | mas_vnr_type | integer | Masonry veneer type (collapse into "None", "BrkFace", "Stone") | Dummified the 3 categories, dropped "None" to set it as baseline |
| 7 | exter_qual | integer | Evaluates the quality of the material on the exterior | 0 (fair) to 3 (excellent) |
| 8 | foundation | integer | Type of foundation (PConc = Poured Concrete, Others = Brick & Tile, Cinder Block, Slab, Stone, Wood | PConc = 1, Others = 0 |
| 9 | bsmt_qual | integer | Evaluates the height of the basement | 0 (none) to 5 (excellent) |
| 10 | bsmt_exposure | integer | Refers to walkout or garden level walls | 0 (none) to 4 (good exposure) |
| 11 | heating_qc | integer | Heating quality and condition | 0 (poor) to 4 (excellent) |
| 12 | central_air | integer | Central air conditioning | Yes = 1, No = 0 |
| 13 | electrical | integer | Electrical system (SBrkr = Standard Circuit Breakers & Romex, All Others = Fuse Box over 60 AMP and all Romex wiring (Average), 60 AMP Fuse Box and mostly Romex wiring (Fair), 60 AMP Fuse Box and mostly knob & tube wiring (poor) and Mixed)| SBrkr = 1, Others = 0 |
| 14 | functional | integer | Home functionality (Typ = Typical Functionality, All others = Minor Deductions 1, Minor Deductions 2, Moderate Deductions, Major Deductions 1, Major Deductions 2, Severely Damaged and Salvage only) | Typ = 1, Others = 0 |
| 15 | fireplace_qu | integer | Fireplace quality | 0 (none) to 5 (excellent) | 
| 16 | garage_finish | integer | Interior finish of the garage | 0 (none) to 3 (finished) |
| 17 | paved_drive | integer | Paved driveway (Y = paved, P = partial pavement, N = dirt/gravel | Y = 1, P and N = 0 |
| 18 | sale_type | integer | Type of sale ("WD" = Warranty Deed - Conventional, "COD" = Court Officer Deed/Estate, "New" = Home just constructed and sold, "Others" = Warranty Deed - Cash, Warranty Deed - VA Loan, Contract 15\% Down payment regular terms, Contract Low Down payment and low interest, Contract Low Interest, Contract Low Down, and Other | Dummified the 4 categories, dropped "WD" to set it as baseline dummy |
| 19 | lot_frontage_imputed | float | Linear feet of street connected to property, with imputation using linear regression on lot area and neighborhood | 21ft to 400ft |
| 20 | year_built | integer | Original construction date | 1872 to 2010 |
| 21 | year_remod/add | integer | Remodel date (same as construction date if no remodeling or additions) | 1950 to 2010 |
| 22 | mas_vnr_area | float | Masonry veneer area in square feet | 0 sqft to 1600 sqft |
| 23 | bsmtfin_sf_1 | float | Basement finished area in square feet | 0 sqft to 5644 sqft |
| 24 | total_bsmt_sf | float | Total square feet of basement area | 0 sqft to 6110 sqft |
| 25 | gr_liv_area | integer | Above grade (ground) living area square feet | 334 sqft to 5642 sqft |
| 26 | full_bath | integer | Full bathrooms above grade | 0 to 4 |
| 27 | garage_area | float | Size of garage in square feet | 0 sqft to 1418 sqft |
| 28 | wood_deck_sf | integer | Wood deck area in square feet | 0 sqft to 1424 sqft |
| 29 | open_porch_sf | integer | Open porch area in square feet | 0 sqft to 547 sqft |


### Modeling
I ran a total of 12 models for comparison. R-squareds and root mean squared errors were used as metrics to gauge model performance. Feature engineering such as transforming continuous variables by using natural logarithmic and square root functions, converting ordinal variables to dummies and using interaction terms were tried.

For example, I found that some continuous variables are more suited for log-transformations while other are better for square-rooting. Below is the table of my selected transformations.

| Variable | Transformation | 
|----------|----------------|
| lot_frontage_imputed | Natural Log |
| mas_vnr_area | Natural Log  |
| bsmtfin_sf_1 | Square Root |
| total_bsmt_sf | Square Root |
| gr_liv_area | Natural Log |
| garage_area | Square Root |
| wood_deck_sf | Natural Log |
| open_porch_sf | Natural Log |
| saleprice | Natural Log |

I found that converting ordinal variables to dummies did not result in a significant improvement in model performance, while potentially introducing overfitting issues. Some of the estimated coefficients do not make sense, and can potentially cause clients to lose confidence in our company. For example, coefficients for overall_qual levels 2 and 3 are lower than overall_qual 1, which imply that a higher quality house is worth less than a lower quality house, all else equal. It is thus better NOT to convert all ordinal values to dummies and to pre-determine the natural ordering of the quality levels before we ask the model to fit the paramters. I also found interaction terms can be a dangerous proposition when dealing with housing variables. While R2s and RMSEs may look deceptively good on train and test sets, the Kaggle scores can be off the charts, which is a sign of overfitting. In the end, I decided to go with a simpler model for my final selection, i.e. 29 chosen variables with 9 of them converted by log and sqrt transformations. Lasso regression with a tuned hyperparamter of alpha = 0.00034 is selected to estimate the coefficients of the features.

### Conclusions and Recommendations
Model results show that gross living area, neighborhood, quality, air conditioning and functionalities are the top 5 factors in driving house prices. Buyers should also look out for houses that are being sold under court-appointed fiduciaries, as they are generally sold at a reasonable discount when compared to another house with equivalent physical attributes.


