from sklearn.tree import DecisionTreeClassifier

def train_decision_tree_model(X_train, y_train):
    # Initialize the Decision Tree classifier
    dt_model = DecisionTreeClassifier(
        criterion='gini',      
        max_depth=None,        
        random_state=42
    )

    # Fit the model on training data
    dt_model.fit(X_train, y_train)
    return dt_model