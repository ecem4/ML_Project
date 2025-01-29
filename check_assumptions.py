#Function nr 2
#Dependent variable: "STAI_Trait_Anxiety" 
#Predictors:"CERQ_SelfBlame", "CERQ_Rumination", "CERQ_Catastrophizing", "COPE_SelfBlame", "LOT_Optimism", "LOT_Pessimism", "PSQ_Worries", "PSQ_Tension"
#The Function: 
#Input: Output of function nr 1
#Output: Checking how strong the correlation between predictors and dependent variable, making boxplots to find and remove outliers, and homogeneity of variance.

#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

'''dep_var = 'STAI_Trait_Anxiety'

datasets_dict = { 'CERQ.csv': ['CERQ_SelfBlame', 'CERQ_Rumination', 'CERQ_Catastrophizing'],
                  'COPE.csv': ['COPE_SelfBlame'],
                  'LOT-R.csv': ['LOT_Optimism', 'LOT_Pessimism'],
                  'PSQ.csv': ['PSQ_Worries', 'PSQ_Tension'],
                  'NEO_FFI.csv': ['NEOFFI_Neuroticism', 'NEOFFI_Extraversion'],
                  'TICS.csv': ['TICS_ChronicWorrying'],
                  'TEIQue-SF.csv': ['TeiQueSF_well_being'],
                  'STAI_G_X2.csv': ['STAI_Trait_Anxiety']}

data_after_preparation = Data_preparation(datasets_dict)'''

'''datasets_dict = { 
    'CERQ.csv': ['CERQ_SelfBlame', 'CERQ_Rumination', 'CERQ_Catastrophizing'],
    'COPE.csv': ['COPE_SelfBlame'],
    'LOT-R.csv': ['LOT_Optimism', 'LOT_Pessimism'],
    'PSQ.csv': ['PSQ_Worries', 'PSQ_Tension'],
    'NEO_FFI.csv': ['NEOFFI_Neuroticism', 'NEOFFI_Extraversion'],
    'STAI_G_X2.csv': ['STAI_Trait_Anxiety']
}'''
#Needed library for homogeneity of variance check imported
from scipy.stats import levene

def check_assumptions(datasets_dict, y):
#Load the data first - remove the detailed code - no need 
    data = datasets_dict 
    
#1_Checking Correlation mAtrix
    correlation_matrix = data.corr()
    correlation_threshold = 0.3  
    dependent_variable = [y]

#Checking correltions of each predictor with the dependent_variable 

    sufficient_correlations = {}
    for col in correlation_matrix.columns:
        sufficient_correlations[col] = {}
        for dep_var in dependent_variable:
            sufficient_correlations[col][dep_var] = abs(correlation_matrix.loc[col, dep_var]) >= correlation_threshold

# Tried to remove predictors with insufficient correlation - data.drop
    predictors_to_remove = [col for col in sufficient_correlations if not sufficient_correlations[col][dependent_variable[0]]]
    data.drop(columns=predictors_to_remove, inplace=True)

# 2_Boxplots
    boxplots = {}
    for col in data.columns:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
# 3_Homogeneity of variance - Levene's test - imported relevant library
    homogeneity_results = {"variable": [],
                           "F value": [],
                           "p_value": []}
    for col in data.columns:
        if col != dependent_variable[0]:
            stat, p_value = levene(data[dependent_variable[0]], data[col])
            homogeneity_results["variable"].append(col)
            homogeneity_results["F value"].append(stat)
            homogeneity_results["p_value"].append(p_value)
            
    homogeneity_results = pd.DataFrame(homogeneity_results)

    correlation_matrix_after_excluding_variables = data.corr()
        
#Visualize 
    correlations_with_anxiety = correlation_matrix[dependent_variable].drop(labels=dependent_variable)

    plt.figure(figsize=(10, 6))
    correlations_with_anxiety.plot(kind='bar', color='skyblue')
    plt.axhline(0.3, color='green', linestyle='--', label='Threshold (0.3)')
    plt.axhline(-0.3, color='red', linestyle='--', label='Threshold (-0.3)')
    plt.title(f"Correlations with {dependent_variable}", fontsize=14)
    plt.xlabel("Predictors", fontsize=12)
    plt.ylabel("Correlation Coefficient", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return correlation_matrix, sufficient_correlations, boxplots, homogeneity_results

#To show a bit more Clear (optional) 

results = check_assumptions(data_to_analyze, "STAI_Trait_Anxiety")

if results:

    # Correlation_matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Sufficient_correlations
    print("\nSufficient Correlations:")
    for predictor, correlations in sufficient_correlations.items():
        print(f"{predictor}: {correlations}")

    # Homogeneity 
    print("\nHomogeneity of Variance (Levene's Test):")
    for col, results in homogeneity_results.items():
        print(f"{col}: Statistic={results['statistic']:.3f}, p-value={results['p_value']:.3f}")

    # Show boxplots- now working 
    for col, boxplot in boxplots.items():
        boxplot.show()







