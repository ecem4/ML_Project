# Predicting Anxiety: The Impact of Negative vs. Positive Thinking #

*The project repository of Machine Learning and Advanced Programming*

## Description ##
Our project aims to investigate whether anxiety levels can be predicted by factors such as negative and positive thinking. We are using data from the Leipzig Mind-Brain-Body (LEMON) dataset to examine how variables related to self-blame, optimism, pessimism, and other psychological factors affect anxiety.

## Research Paper and Dataset ##
The project is based on the LEMON dataset as presented in the paper:

 Paper:  https://www.nature.com/articles/sdata2018308
 
 Dataset access: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/

## Research question ##
     Can we predict anxiety by examining positive and negative thinking patterns? 
   
* dependent variable: STAI_Trait_Anxiety (Trait anxiety score from the State-Trait Anxiety Inventory).

* predictors:
   - COPE_SelfBlame
   - CERQ_SelfBlame
   - CERQ_Rumination
   - CERQ_Catastrophizing
   - LOT_Optimism
   - LOT_Pessimism
   - PSQ_Worries
   - PSQ_Tension
   - NEOFFI_Neuroticism
   - NEOFFI_Extraversion

## Required libraries:

```pandas```
```scikit-learn```
```seaborn```
```matplotlib```

You can install them from the requirements.txt file. Use the following command to install them:
```pip install -r requirements.txt```

## Data Preprocessing and exploration ##

1. Cleaning: We remove missing values.
2. Normalization: We normalize the data using StandardScaler from scikit-learn.
3. Exploration: Summary statistics
4. Correlation Matrix: We generate a correlation matrix to identify which variables strongly correlate with the dependent variable (STAI_Trait_Anxiety).
4. Factor Analysis: If two or more predictors strongly correlate with each other, we merge them using factor analysis.
5. Visualization: We create boxplots for each predictor to identify and exclude outliers.
   
## Functions ## 

1. ```Data_preperation``` function

Input: A dictionary where keys are chosen datasets, and values are specific columns.

Output: A dataset with chosen variables from input datasets that are normalized, and missing data are excluded.

The function does the following:
- Merges datasets from the input dictionary.
- Excludes missing data.
- Normalizes the variables using StandardScaler.

2. ```check_assumptions``` function

Input: Output of function nr 1

Output: Matrix of correlation with information which correlations with depended variables are sufficient; boxplots of each variable.

Function make correlation matrix and list of coeffs of correlations and loop go through the list and create dict in which each key is name of variable and value is information if predictor has enough coeff. Moreover plot boxplots. 

## Modeling ##

Supervised Learning with Multiple Regression Model to predict the anxiety score (STAI_Trait_Anxiety).

1. Train/Test Split: The dataset is split into training and testing sets.
2. Model Training: A linear regression model is trained using the selected predictors.
3. Evaluation: The model is evaluated using metrics like R-squared to assess how well the predictors explain variance in anxiety scores.

## Conclusion ##

The project explores whether negative or positive thinking can predict anxiety levels. By preprocessing the LEMON dataset and building a regression model, we aim to identify the strongest predictors of anxiety. The results will help us understand the psychological factors associated with anxiety and whether interventions targeting these thought patterns could mitigate anxiety.



