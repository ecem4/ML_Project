#Function nr 2
#Dependent variable: "STAI_Trait_Anxiety" 
#Predictors:"CERQ_SelfBlame", "CERQ_Rumination", "CERQ_Catastrophizing", "COPE_SelfBlame", "LOT_Optimism", "LOT_Pessimism", "PSQ_Worries", "PSQ_Tension"
#The Function: 
#Input: Output of function nr 1
#Output: Checking how strong the correlation between predictors and dependent variable, making boxplots to find and remove outliers, and homogeneity of variance.

datasets_dict = { 
    'CERQ.csv': ['CERQ_SelfBlame', 'CERQ_Rumination', 'CERQ_Catastrophizing'],
    'COPE.csv': ['COPE_SelfBlame'],
    'LOT-R.csv': ['LOT_Optimism', 'LOT_Pessimism'],
    'PSQ.csv': ['PSQ_Worries', 'PSQ_Tension'],
    'NEO_FFI.csv': ['NEOFFI_Neuroticism', 'NEOFFI_Extraversion'],
    'STAI_G_X2.csv': ['STAI_Trait_Anxiety']
}
def check_assumptions(datasets_dict):

#1_Checking Correlation mAtrix
    correlation_matrix = datasets_dict.corr()
    sufficient_correlations = {}
    boxplots = {}
    homogeneity_results = {}

#setting the threshold 0.3

    correlation_threshold = 0.3  
    dependent_variable = ["STAI_Trait_Anxiety"]

#Checking correltions of each predictor with the dependent_variable 

    for col in correlation_matrix.columns:
        sufficient_correlations[col] = {}
        for dep_var in dependent_variable:
            if abs(correlation_matrix.loc[col, dep_var]) >= correlation_threshold:
                sufficient_correlations[col][dep_var] = True
            else:
                sufficient_correlations[col][dep_var] = False

print(correlation_matrix)
print(sufficient_correlations)

#Visualize to see which predictor to exclude
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

#So for those results we need to remove the predictor: COPE_SelfBlame, PSQ_Worries, PSQ_Tension and keeping the rest
