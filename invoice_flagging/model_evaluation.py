from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_tuned_random_forest(X_train, y_train):
    """Trains a Random Forest using GridSearch to find the best parameters."""
    # These are the parameters tested in the video
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 4, 5],
        'min_samples_split': [2, 5],
        'criterion': ['gini']
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Use GridSearch to test all combinations above
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
    
    print("\nRunning GridSearchCV (this might take a few seconds)...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    # Return the absolute best version of the model
    return grid_search.best_estimator_

def evaluate_classifier(model, X_test, y_test, model_name="Random Forest"):
    """Predicts on test data and prints classification metrics."""
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return acc