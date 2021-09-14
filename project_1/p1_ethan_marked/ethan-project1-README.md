# Project 1: Standardized Test Analysis
## Race and Standardised Tests, by Ethan Leow

### Hypothetical Background and Problem Statement
Black Lives Matter (BLM) is a political and social movement protesting against incidents of police brutality against black people. We are a non-profit student organisation that is an offshoot of BLM, and our main goal is to voice support against institutionalised odds that are stacked against blacks and other minority groups.

We wish to investigate and gather evidence that standardised tests such as SAT and ACT are discriminatory against blacks and other non-white minority groups. By conducting some basic exploratory data analysis, we hope to observe some negative relationships between test scores and minority races, while anticpating and potentially answering some criticisms or skepticisms from donors.

The hope is to use the results of our EDA to convince these donors to fund an in-depth study to investigate if minority students from comparable socioeconomic backgrounds score lower than their White peers.

### Data Dictionary
1. ACT and SAT data from General Assembly
2. Racial demographics data (by state) from Kaiser Foundation
    - (https://www.kff.org/other/state-indicator/distribution-by-raceethnicity/?currentTimeframe=2&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D)
3. Household income data (by state) from US Census
    - (https://data.census.gov/cedsci/table?q=S1901&g=0100000US.04000.001&tid=ACSST1Y2019.S1901&hidePreview=true&tp=true)
    - See US Census metadata file in the data folder for more info on the data categories
4. Gini coefficient data (by state) from US Census
    - (https://data.census.gov/cedsci/table?q=B19083&g=0100000US.04000.001&tid=ACSDT1Y2018.B19083&tp=true&hidePreview=false)

|Feature|Type|Original Source|Description|
|---|---|---|---|
|state|object|Common to all data sources|Name of states in the U.S.A. (excludes Puerto Rico and District of Columbia)| 
|year|integer|Common to all data sources|Calendar year (Values: 2017 and 2018 only)| 
|act_participate|float|General Assembly (act_2017.csv, act_2018.csv)|ACT participation rate in the state (Values: 0.00-1.00)| 
|act_english|float|General Assembly (act_2017.csv, act_2018.csv)|Average ACT English score in the state (Values: 1-36, data available for the Year 2017 only)| 
|act_math|float|General Assembly (act_2017.csv, act_2018.csv)|Average ACT Math score in the state (Values: 1-36, data available for the Year 2017 only)| 
|act_reading|float|General Assembly (act_2017.csv, act_2018.csv)|Average ACT Reading score in the state (Values: 1-36, data available for the Year 2017 only)| 
|act_science|float|General Assembly (act_2017.csv, act_2018.csv)|Average ACT Science score in the state (Values: 1-36, data available for the Year 2017 only)| 
|act_composite|float|General Assembly (act_2017.csv, act_2018.csv)|Average ACT Composite score in the state (Values: 1-36)| 
|sat_participate|float|General Assembly (sat_2017.csv, sat_2018.csv)|SAT participation rate in the state (Values: 0.00-1.00)| 
|sat_english|integer|General Assembly (sat_2017.csv, sat_2018.csv)|Average SAT Evidence-Based Reading and Writing score in the state (Values: 200-800)| 
|sat_math|integer|General Assembly (sat_2017.csv, sat_2018.csv)|Average SAT Math score in the state (Values: 200-800)| 
|sat_total|integer|General Assembly (sat_2017.csv, sat_2018.csv)|Average SAT Total score in the state (Values: 400-1600)| 
|median_income|integer|US Census Bureau (S1901_2017.csv, S1901_2018.csv)|Median household income in the state (dollars)| 
|mean_income|integer|US Census Bureau (S1901_2017.csv, S1901_2018.csv)|Mean household income in the state (dollars)| 
|gini|float|US Census Bureau (B19083_2017.csv, B19083_2018.csv)|Gini index of income inequality in the state (Values: 0.00-1.00, where 0 == complete equality and 1 == complete inequality)| 
|white|float|Kaiser Family Foundation (race_2017.csv, race_2018.csv)|Percentage of a state's population that identifies as belonging to the White race| 
|black|float|Kaiser Family Foundation (race_2017.csv, race_2018.csv)|Percentage of a state's population that identifies as Black or African American| 
|hispanic|float|Kaiser Family Foundation (race_2017.csv, race_2018.csv)|Percentage of a state's population that identifies as Hispanic or Latino| 
|asian|float|Kaiser Family Foundation (race_2017.csv, race_2018.csv)|Percentage of a state's population that identifies as Asian| 
|race_others|float|Kaiser Family Foundation (race_2017.csv, race_2018.csv)|Percentage of a state's population that identifies as American Indian, Native Hawaiian, Pacific Islander and all others| 


### Brief Summary of Analysis
I start by cleaning the ACT and SAT data. Errors were found in 2017's ACT Science scores (Maryland), 2018 ACT file where Maine's data was repeated twice and "District of Columbia" has a small typo. An error was also found in 2017's SAT Math scores.

No issues were found with US Census' economic data and Kaiser Family Foundation's racial data.

All 8 datasets were combined into a Pandas dataframe called master_df. One observation is that different data sources treat Puerto Rico ("PR") and District of Columbia ("DC") differently. Technically they are not U.S. states (though liberal Democrats have been lobbying to make them states to tilt the Senate in their favour!), but the ACT/SAT dataset has DC but no PR, whereas Kaiser's dataset has PR but no DC. US Census has both. To make it easier for apples-to-apples comparison when combining datasets, I drop PR and DC, leaving only 50 states in the master_df file.

A correlation heatmap was created to analyse all data variables. Some highly correlated variables were dropped as they do not provide extra information beyond what other variables can provide. They are:
* ACT Math, Reading and Science scores for 2017
* SAT subject scores
* Median household income

A new correlation heatmap and sns.pairplot are drawn up to analyse relationships between all remaining variables. Next, I create mulitple scatter plots with fitted lines to look into my variables of interest in further details.

I see that ACT and SAT state scores seem to be lower in states with high propoportion of minority races.
* In particular, states with high proportion of Blacks and Native Americans seem to do poorly in ACT.
* States with high proportion of Asians and Hispanics seem to do poorly in SAT.

To address a potential criticism that White folks are more hardworking than the minorities and hence earn higher income to give their children a better education, leading to higher SAT/ACT scores, I plot the relationship between state mean income level and race.
* States with high proportion of Whites actually have lower income levels!

I also plot SAT/ACT scores against income level. If higher income leads to better educational resources, and better education leads to better standardised test scores, the charts show otherwise. SAT scores seem to be lower in high-income states.

### Conclusions and Recommendations
I show that exploratory data analysis of state test scores and state racial proportions seems to suggest a negative relationship between scores and minority races. States with higher proportion of Whites seem to do better in standardised tests. I also show that the fact that "white" states perform better is not due to higher average income levels in those states. In fact, income has mixed results when compared to standardised test results.

My recommendation is that we take these promising results and approach donors to seek more funding to conduct an in-depth study to investigate if minority students from comparable socioeconomic backgrounds score lower than their White peers.