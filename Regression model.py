dep_var = 'STAI_Trait_Anxiety'

datasets_dict = { 'CERQ.csv': ['CERQ_SelfBlame', 'CERQ_Rumination', 'CERQ_Catastrophizing'],'COPE.csv': ['COPE_SelfBlame'],
'LOT-R.csv': ['LOT_Optimism', 'LOT_Pessimism'],
    'PSQ.csv': ['PSQ_Worries', 'PSQ_Tension'],
    'NEO_FFI.csv': ['NEOFFI_Neuroticism', 'NEOFFI_Extraversion'],
    'STAI_G_X2.csv': ['STAI_Trait_Anxiety']}

data_after_preparation = Data_preparation(datasets_dict)

data_after_checking_assumptions = check_assumptions(data_after_preparation, dep_var)[0]


X = data_after_checking_assumptions.drop(dep_var, axis=1)
y = data_after_checking_assumptions[dep_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print(f'Intercept: {reg_model.intercept_}')
print(list(zip(X, reg_model.coef_)))
y_pred= reg_model.predict(X_test)  
x_pred= reg_model.predict(X_train) 
print("Prediction for test set: {}".format(y_pred))

reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_model_diff

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)
