#First model

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
