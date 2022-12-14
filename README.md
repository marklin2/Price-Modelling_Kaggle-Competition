# Price-Modelling_Kaggle-Competition
## Intro
The basic conept of the competition is to create models to predict the price of Airbnb listings in NY.

Before digging into my data cleaning and prediction process, I would like to briefly explain why I found that the Kaggle competition has benefited my academic and professional career. I intend to become a player evaluation analyst in Major League Baseball, particularly for discovering young and overlooked talents, my responsibilities would include using R to create models and prepare data-driven reports to assess players’ abilities. As modern baseball is a combination of old school scouting and analytics, scouting reports are created by both traditional baseball language (e.g. This kid has enough raw juice to hit 30 bombs a year) and numbers (e.g. The pitcher has a 60 future value of his baseball), which I found parallel with out Kaggle Airbnb datasets. I will narrate further details and draw the connections I found fascinating in the rest of the report. <br/>


## Data Exploration
After reading the Airbnb datasets into R, I applied str() function to have a basic sense of all the variables and the corresponding datatypes.

## Data Preparation
In this section, I will separate my data cleaning process mainly into three parts, “Converting numeric values into categorical variables”, “Creating text variables by discovering patterns from comments,” and “Creating dummy variables.” <br/>

First of all, as we discussed in class, numerical variables including but not limited to bedroom, bathroom and beds are far from normally distributed, which are skewed towards a side. To effectively utilize the data in further predictions, it is critical to convert the data to categorical variables to enhance the accuracy. For instance, the majority of bedroom data turns out to be either 0 or 1. On top of that, there are less than 10 percent of bedroom data points are greater than 5. The extreme left skewness of the dataset would significant downgrade the predictability of the variable, which is the reason why I translated the data in to bedroom_categorical. I first assigned the new variable to an empty “ ”, and employed mutate function to create new categories in both original train data and the test data. I categorized it as bedroom_0, bedroom_1, bedroom_2, bedroom_greater3, which I believe more accurately specifies the feature of the data.
Moreover, this data cleaning technique also applies to my research in baseball. Recently there is a common statistic that takes a holistic approach on player, which is WAR(Wins Above Replacement). If a player has a 3.5 in WAR, it refers to having this player on your team would give you 3.5 more wins in a season. Most major league players have an WAR between 1.0-2.0; “All Star players” have an WAR above 3; and “Superstars” generally have a WAR north of 5. <br/>


By applying the technique, I am able to effectively separate the WAR by assorted player caliber and increase future effectiveness of predictions.<br/>

Second, I created variables by discovering the patterns from the comments and descriptions. For instance, words including luxury, close, cozy and ideal were highly used in the “Descriptions” column and all had positive effects on the prediction. By using the grep() function, I created a binary variable, with 0 assigned to rows without the text, while 1 assigned to rows including the text. This specific data cleaning technique would allow me to better analyze scouting reports of players in baseball, which specific words such as, athletic, and strong may appear as a plus in the overall assessment.<br/>

Last but not least, to comply with gradient boosting, I transformed the categorical variables to dummy variables. For example, variables including property type, neighborhood group cleansed and room type are transformed into dummy variables.

## Analysis Techniques Applied
In this section, I will discuss three statistical modeling techniques I applied throughout the Kaggle competition, including “Linear Regression Models”, “Decision Trees” and “eXtreme Gradient Boosting”.<br/>

For starters, I adopted linear regression model to test the water for the correlations between the variables and price. I started with an inclusion of variables that I intuitively feel like have a direct impact on the price, including zip code, accommodates, bedroom, beds, bathroom and property type. I further used Feature selection techniques to filter out the best subset out of dozens of variables, which I had the best RMSE that sits around 69. <br/>

Furthermore, I used decision trees, specifically regression, to optimize the model better. Although the RMSE for training datasets have been low and decent, there is a large difference with the RMSE for the test dataset, which the model would be relatively less effective. Therefore, I further applied boosting models to attempt to generate better predictions.<br/>

Last but not least, I created an eXtreme Gradient Boosting model to generate predictions. I first started directly using xgBoost models, which I neglected the cross validation process an ended up obtaining an extreme low RMSE (5.3) for the train data, but recorded a rather high RMSE (72) for the test data. The overfitting issue for gradient boosting have been notorious in making predictions, to tackle the issue, I first dummy coded the variables, and used xgb.cv() function to generate tune rounds. Afterwards, I selected the best tune rounds by filtering the lowest RMSE, and further load the model into xgboost.
