from sklearn.linear_model import LinearRegression
from pandas import read_csv, Series
from numpy import array

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = read_csv('bmi_and_life_expectancy.csv')

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
linear_reg_model = LinearRegression()
X = array(bmi_life_data[['BMI']]).reshape(1, -1)
y = array(bmi_life_data[['Life Expectancy']]).reshape(1, -1)
bmi_life_model = linear_reg_model.fit(X, y)


# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)
