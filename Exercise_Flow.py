from Classes import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


X, y = load_boston(return_X_y=True)

# p is the number of features (columns) in the dataset and n is the number of samples (rows) that we have.
# So the dimension of X is nxp.
n, p = X.shape
print(f'p = {p}, n = {n}')

# Fit the OLS model and calculating the training MSE
model_ols = Ols()
model_ols._fit(X, y)
training_mse = model_ols.score(X,y)
print(f'The training MSE is: {training_mse}')

# Plot a scatter plot
y_pred = model_ols._predict(X)
plt.scatter(y, y_pred, alpha=0.5)
plt.title("Predicted vs. Actual - OLS Model")
plt.xlabel('Y True')
plt.ylabel('Y Predicted ($\hat{Y}_{OLS}$)')

shuffle_split = ShuffleSplit(n_splits=20, test_size=.25, random_state=42)

all_training_mse = []
all_test_mse = []

# Split the data to 75% train and 25% test 20 times
for train_index, test_index in shuffle_split.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_ols = Ols()
    model_ols._fit(X_train, y_train)
    all_training_mse.append(model_ols.score(X_train, y_train))
    all_test_mse.append(model_ols.score(X_test, y_test))

print(f'The average MSE for train is: {round(np.mean(all_training_mse), 3)}')
print(f'The average MSE for test is: {round(np.mean(all_test_mse), 3)}')

# Use a t-test to prove that the MSE for training is significantly smaller than for testing.
t_test = stats.ttest_rel(all_training_mse, all_test_mse, alternative='less')
print(f'The p-value is: {t_test[1]}')
print(f'The t-statistic is: {t_test[0]}')
print("\nP-value is less than 0.05, so we reject the null hypothesis and say in confidence of 95% that the two MSE are not the same")

# Run the OLS Gradient Descent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
ols_gd = OlsGd()
ols_gd._fit(X_train, y_train)
print(f'\nThe MSE of the test set is: {ols_gd.score(X_test, y_test)}')

# Plot the loss function convergence
plt.plot(ols_gd.loss)
plt.title("Loss convergance of OLS Gradient descent")
plt.xlabel("Iterations")
plt.ylabel("Loss")

# plot scatter plot
plt.scatter(y_test, ols_gd._predict(X_test), alpha=0.5)
plt.xlabel('Y True')
plt.ylabel('Y Predicted ($\hat{Y}_{GD}$)')
plt.title("Predicted vs. Actual Values of OLS Gradient Descent")

# learning rate plot
alpahs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

for learning_rate in alpahs:
    ols_gd = OlsGd(learning_rate=learning_rate, verbose=False)
    ols_gd._fit(X_train, y_train)
    results = ols_gd._predict(X_test)
    plt.plot(ols_gd.loss, label='Learning Rate: ' + str(learning_rate))
    plt.legend()

# run our implementations for OLS Ridge and GD Ridge
Ridge = RidgeLs(ridge_lambda=0.1)
Ridge._fit(X_train, y_train)
print(f'The OLS Ridge MSE for test is {round(Ridge.score(X_test, y_test),3)}')

Ridge_GD = RidgeLs_Gd(ridge_lambda=0.1, verbose=False)
Ridge_GD._fit(X_train, y_train)
print(f"The OLS Ridge GD MSE for test is: {round(Ridge_GD.score(X_test, y_test),3)}")

# use scikitlearn implementation for OLS, Ridge and Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

linear_reg = LinearRegression().fit(X_train, y_train)
y_linear_pred = linear_reg.predict(X_test)
print(f'Linear Regression MSE: {mean_squared_error(y_test, linear_reg.predict(X_test))}')

ridge_reg = Ridge(alpha=1.0).fit(X_train, y_train)
y_ridge_pred = linear_reg.predict(X_test)
print(f'Ridge Regression MSE: {mean_squared_error(y_test, ridge_reg.predict(X_test))}')

lasso_reg = Lasso(alpha=1.0).fit(X_train, y_train)
y_lasso_pred = linear_reg.predict(X_test)
print(f'Lasso Regression MSE: {mean_squared_error(y_test, lasso_reg.predict(X_test))}')
#