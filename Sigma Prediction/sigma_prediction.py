import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

# Load the Excel file into a Pandas DataFrame
df = pd.read_excel("C:/Users/S D Rao/Documents/ML_Project/data.xlsx")

# CORRELATION
# Scatter Plot with Regression Line
def cor_reg(x,y):
    
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x, y=y, data=df, scatter_kws={'s': 10})
    plt.title('Scatter Plot with Regression Line')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

#print(cor_reg("CONC","CONC"))
#print(cor_reg("CONC","FREQ"))
#print(cor_reg("CONC","S"))
#print(cor_reg("CONC","SP"))
#print(cor_reg("FREQ","FREQ"))
#print(cor_reg("FREQ","S"))
#print(cor_reg("FREQ","SP"))
#print(cor_reg("S","S"))
#print(cor_reg("S","SP"))


# NORMALIZATION
columns_to_normalize = ["FREQ", "S", "SP"]

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Apply Min-Max normalization to the specified columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Specify the columns you want to use for prediction
features = ["CONC", "FREQ", "S", "SP"]  # Add other features as needed
target = ["S", "SP"]  # Replace with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# SVM with MultiOutputRegressor
svm_model = MultiOutputRegressor(SVR())
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation Metrics
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r_squared = r2_score(y_true, y_pred)

    print(f"Metrics for {model_name}:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Value: {r_squared:.4f}")
    print("")

# Evaluate each model
evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_dt, "Decision Tree")
evaluate_model(y_test, y_pred_svm, "SVM")
evaluate_model(y_test, y_pred_rf, "Random Forest")