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

## Data Preprocessing and exploration ##

## Functions ## 
1. 'Data_preperation' function
Input: A dictionary where keys are chosen datasets (as file paths), and values are specific columns (i.e., predictors).
Output: A dataset with chosen variables from input datasets that are normalized, and missing data are excluded.
The function does the following:
Merges datasets from the input dictionary.
Excludes missing data.
Normalizes the variables using StandardScaler.

3. 'check_assumptions' function



## Modeling ##



