# Project 3 - Web APIs & NLP
## by Ethan Leow


### Problem Statement
I am a newbie to the marathon scene, having completed my first virtual StanChart marathon when the country was locked down during Covid-19 last year. I am itching to do my first ultramarathon (defined as any run beyond 42.195km) and have prepared a series of posts to ask running veterans for training advice. I found that the most popular running community to be r/running with 1.5 million members but most of their posts are for running newbies or people looking to cover their first 10k / 10mile, or their first marathon. The two next most popular reddit communities for more advanced types of running would be r\ultrarunning and r\AdvancedRunning. These are some of their attributes:

|subreddit|Date Created|Number of Members|Members Online (at time of checking)|
|---------|------------|-----------------|------------------------------------|
|r\ultrarunning|Feb 4, 2012|45,900|31 (on 27 Sep, 1700h, SG time)|
|r\AdvancedRunning|Oct 31, 2011|142,000|169 (on 27 Sep, 1700h, SG time)|

A quick scan of their recent posts is not very helpful. Both communities have a lot of posts about how to train for 50k and beyond, so it is not easy to differentiate which post should belong to which subreddit community:
- For example, this is a post from r\ultrarunning from yesterday: "Should I run my second ultra in 7 weeks? I just finished my first ultra this Saturday (Sept 25) in 8 hours (50Km)....I'm debating to do an ultra, either 60Km or 50 Miles (80Km) on November 13th. That will be the last race of the year." 
- And this is a post from r\AdvancedRunning from yesterday: "Last week I completed my first 100 mile race. It has been a goal of mine for the last few years... I hope to use this race as a stepping stone for longer and harder miles. There is truly something addicting about completing such an extraordinary goal."

It is time to deploy my data science skills to come up with an algorithm to decide which running community most suited for my questions...

### Data Dictionary
That data is extracted from Pushshift Reddit API, created by the \r\datasets mod team to help provide enhanced functionality and search capabilities for searching Reddit comments and submissions. As my problem statement is to figure out where to post my submmision, I only searched for Reddit submmission entries and left out comments. I did not extract metadata and only scrapped for the following: 

| No. | Feature | Type |  Description | Remarks |
|-----|---------|------|-------------|---------|
| 1 | subreddit | String| Name of specific subreddit | "ultrarunning", "AdvancedRunning" |
| 2 | author | String | Name of submission's author |  | 
| 3 | title | String | Title of submission | |
| 4 | selftext | String | Body content of submission | |
| 5 | created_utc | Integer | Time of creation in UTC terms | |

### Data Scrapping 
About 20% of the posts have null or \[removed\] as the full body content, so my scrapping function filters them out at the scrapping stage. I also filtered out reposts by authors. I download 100 posts per request, with a random sleep time of 1 to 3 second between each request. I set a total limit for 2500 posts per subreddit, as I notice that I usually get a 502 status error from Pushshift's server when my scrapper goes beyond that limit. 

### Data Cleaning
Posts that are URLs only, or mostly URLs (e.g. 2-3 words + URL) are removed. I also removed posts that seem to be written by AI moderators. Common English stop words (from NLTK library), common overlapping words in both subreddits, unique stop words that can identify the subreddit e.g. "ultrarunning" or "advancedrunning" taken out from the posts.

### Modeling
I ran a total of 22 permutations for comparison sake. Accuracy scores (mean 5-fold CV score on train set, and accuracy score on test set) are used as the main metric for comparison. Secondary metrics such as sensitivity, specificity, precision and runtime are calculated in case of a tie in accuracy scores.
The word vectorizers I use are Count Vectorizer and TD-IDF Vectorizer. The base estimators I use are Multinomial Naive Bayes, Logistic, Random Forest, Extra Trees, and towards the end a small section on AdaBoost (on a shallow DecisionTreeClassifier) and XGBoost as those were only taught after I finished my project. All models were tuned using GridSearchCV at all stages, so they may take a while to run. After I selected my final model to use, I did a few more rounds of hyperparameter-tuning to explore a wider range of possibilities for my paramater choices until I could not improve the scores further. 

### Reflections and Conclusions 
Before the project started, I naively thought that well-regarded non-parametric models in Kaggle contests such as RandomForest would do the best. That was my baseline assumption.

Within the parametric space, I am pleasantly surprised to see that a simple model such as Logistic Regression outpeforming a more complicated model like Multinomial Naive Bayes. When it didn't, my thought is that the independence condition among word features is probably grossly violated since real data sets are never perfectly independent, so it can't perform as well. A [paper](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) by [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng) (we watched one of his videos in class) and [Michael Jordan](https://en.wikipedia.org/wiki/Michael_I._Jordan) (Andrew's PhD advisor, not the Chicago Bulls legend) states that logistic regression should perform better than Naive Bayes as training size gets larger, though I am not sure if my sample size is considered 'large' enough.

When I moved to the non-parametric space, I am also pleasantly surprised to see that RandomForest and even popular techniques such as XGBoost couldn't outperform Logistic Regression. I can't find any papers online to explain why this is the case, but the old-fashioned person in me instinctively says that if I can achieve the same results using a simple and easy-to-understand model, we should go for it as it reduces the risks of overfitting and it is easier to dissect if things go wrong.

