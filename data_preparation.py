#Function nr 2
#Dependent variable: "STAI_Trait_Anxiety" 
#Predictors:"CERQ_SelfBlame", "CERQ_Rumination", "CERQ_Catastrophizing", "COPE_SelfBlame", "LOT_Optimism", "LOT_Pessimism", "PSQ_Worries", "PSQ_Tension"
#The Function: 
#Input: Output of function nr 1
#Output: Checking how strong the correlation between predictors and dependent variable, making boxplots to find and remove outliers, and homogeneity of variance.

def check_assumptions(data):

#1_Checking Correlation mAtrix
    correlation_matrix = data.corr()
    sufficient_correlations = {}
    boxplots = {}
    homogeneity_results = {}

#setting the threshold 0.3

    correlation_threshold = 0.3  
    dependent_variable = ["STAI_Trait_Anxiety"]

#Checking correltions of each predicotr with the dependent_variable 

    for col in correlation_matrix.columns:
        sufficient_correlations[col] = {}
        for dep_var in dependent_variable:
            if abs(correlation_matrix.loc[col, dep_var]) >= correlation_threshold:
                sufficient_correlations[col][dep_var] = True
            else:
                sufficient_correlations[col][dep_var] = False

print(correlation_matrix)
print(sufficient_correlations)

#So for those results we need to remove the predictor: COPE_SelfBlame, PSQ_Worries, PSQ_Tension and keeping the rest
