from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42, max_depth=4)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42, max_depth=4)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Predicts on test data and prints performance metrics."""
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print(f"\n--- {model_name} Performance ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"MSE:      {mse:.4f}")
    
    return r2