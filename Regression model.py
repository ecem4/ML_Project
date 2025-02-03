import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
import numpy as np
from sklearn import metrics
from sklearn.model_selection import learning_curve

'''''Function nr 1 :  Data_preparation
Input: Dictionary where keys are chosen datasets and values are specific columns
Output: Dataset with chosen variables form input datasets which are normalized and lack of data are excluded. 
loop in the function will go through keys of input dict and values of each keys to import data; emerge them in one dataset. 
After that next loop will exclude lack of data and data will be normalized '''''

def Data_preparation(datasets_dict):
    merged_df = pd.DataFrame()  # This is an empty dataframe to hold the merged data from all datasets
    
    # Iterate through the dictionary, read each dataset, select specified columns, and merge them
    for dataset, columns in datasets_dict.items():
        # Import dataset
        df = pd.read_csv(dataset)
        
        # Select specific columns
        df = df[columns].apply(pd.to_numeric, errors='coerce')
        # Merge with the main dataframe (concatenate along axis=1)
        merged_df = pd.concat([merged_df, df], axis=1)
    
    # Remove rows with missing values (NaN) to ensure the dataset is complete before further processing
    merged_df.dropna(inplace=True)   

    # Standardize (normalize) the data using StandardScaler
    scaler = StandardScaler()
    standardized_data = pd.DataFrame(scaler.fit_transform(merged_df), columns=merged_df.columns)
    
    return standardized_data

dep_var = 'STAI_Trait_Anxiety'

datasets_dict = { 'CERQ.csv': ['CERQ_SelfBlame', 'CERQ_Rumination', 'CERQ_Catastrophizing'],
                  'COPE.csv': ['COPE_SelfBlame'],
                  'LOT-R.csv': ['LOT_Optimism', 'LOT_Pessimism'],
                  'PSQ.csv': ['PSQ_Worries', 'PSQ_Tension'],
                  'NEO_FFI.csv': ['NEOFFI_Neuroticism', 'NEOFFI_Extraversion'],
                  'TICS.csv': ['TICS_ChronicWorrying'],
                  'TEIQue-SF.csv': ['TeiQueSF_well_being'],
                  'STAI_G_X2.csv': ['STAI_Trait_Anxiety']}

data_after_preparation = Data_preparation(datasets_dict)

'''Funcion number 2'''

def check_assumptions(datasets_dict, y):
#Load the data first
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

# 3_Homogeneity of variance - levene's test - imported relevant library
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

    return data, correlation_matrix, correlation_matrix_after_excluding_variables, homogeneity_results

  X = data_after_preparation.drop(dep_var, axis=1)
y = data_after_preparation[dep_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

reg_model_1 = LinearRegression()
fit_reg_model = reg_model_1.fit(X_train, y_train)
print(f'Intercept: {reg_model_1.intercept_}') # point where regression line cut y line
print(list(zip(X, reg_model_1.coef_))) # R-square between each predictior and depended variable
y_pred= reg_model_1.predict(X_test)
x_pred= reg_model_1.predict(X_train)
print("Prediction for test set: {}".format(y_pred))

reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_model_diff

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)

train_sizes, train_scores, test_scores = learning_curve(reg_model_1, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(-train_scores, axis=1)
train_std = np.std(-train_scores, axis=1)
test_mean = np.mean(-test_scores, axis=1)
test_std = np.std(-test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, test_mean, label='validation')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#Second model
data_after_checking_assumptions = check_assumptions(data_after_preparation, dep_var)

X = data_after_checking_assumptions[0].drop(dep_var, axis=1)
y = data_after_checking_assumptions[0][dep_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

reg_model_2 = LinearRegression()
fit_reg_model = reg_model_2.fit(X_train, y_train)
print(f'Intercept: {reg_model_2.intercept_}') # point where regression line cut y line
print(list(zip(X, reg_model_2.coef_))) # R-square between each predictior and depended variable
y_pred= reg_model_2.predict(X_test)
x_pred= reg_model_2.predict(X_train)
print("Prediction for test set: {}".format(y_pred))

reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_model_diff

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)

train_sizes, train_scores, test_scores = learning_curve(reg_model_2, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(-train_scores, axis=1)
train_std = np.std(-train_scores, axis=1)
test_mean = np.mean(-test_scores, axis=1)
test_std = np.std(-test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, test_mean, label='Validation')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#Lasso normalisation 
scores = []
for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    scores.append(lasso_model.score(X_test, y_test))

X_lasso = data_after_checking_assumptions[0].drop(dep_var, axis=1)
y_lasso = data_after_checking_assumptions[0][dep_var]
X_lasso_train, X_lasso_test, y_lasso_train, y_lasso_test = train_test_split(X_lasso, y_lasso, test_size = 0.3, random_state = 100)

names = data_after_checking_assumptions[0].drop(dep_var, axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X_lasso_train, y_lasso_train).coef_
Y_pred = lasso.predict(X_lasso_test)

plt.bar(names, lasso_coef)
plt.xticks(rotation=45)

mae = metrics.mean_absolute_error(y_lasso_test, Y_pred)
mse = metrics.mean_squared_error(y_lasso_test, Y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_lasso_test, Y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)

train_sizes, train_scores, test_scores = learning_curve(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(-train_scores, axis=1)
train_std = np.std(-train_scores, axis=1)
test_mean = np.mean(-test_scores, axis=1)
test_std = np.std(-test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, test_mean, label='Validation')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.grid(True)
plt.show()