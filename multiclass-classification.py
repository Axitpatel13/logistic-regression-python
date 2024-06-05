import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Set threshold for binary classification
threshold = 300000
df['price_above_threshold'] = (df['price'] > threshold).astype(int)

# Define features and target variable
features = ['size', 'bedrooms', 'age']
X = df[features]
y = df['price_above_threshold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Select new data for prediction from the original dataset (example: first 5 rows)
new_data = df[features].head(5)

# Predicting the target for new data
new_predictions = model.predict(new_data)
new_predictions_proba = model.predict_proba(new_data)

print("New Data Predictions (0 = Below threshold, 1 = Above threshold):", new_predictions)
print("New Data Prediction Probabilities:\n", new_predictions_proba)
```

### Example for Multiclass Classification

If you want to classify house prices into multiple categories (e.g., low, medium, high), you can modify the example accordingly:

1. **Load and preprocess the dataset**:
    ```python
    df = pd.read_csv('house_prices.csv')

    # Create multiple categories for classification
    bins = [0, 200000, 400000, np.inf]
    labels = ['low', 'medium', 'high']
    df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)
    
    # Convert categories to numerical values
    df['price_category'] = df['price_category'].cat.codes

    # Define features and target variable
    features = ['size', 'bedrooms', 'age']
    X = df[features]
    y = df['price_category']
    ```

2. **Split the dataset into training and testing sets**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```

3. **Create and train the logistic regression model**:
    ```python
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    ```

4. **Make predictions on the test set and evaluate the model**:
    ```python
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    ```

5. **Predict on a subset of the original data**:
    ```python
    new_data = df[features].head(5)
    new_predictions = model.predict(new_data)
    new_predictions_proba = model.predict_proba(new_data)

    print("New Data Predictions (0 = low, 1 = medium, 2 = high):", new_predictions)
    print("New Data Prediction Probabilities:\n", new_predictions_proba)
    ```

### Complete Code for Multiclass Classification

Here’s the entire code for multiclass classification:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Create multiple categories for classification
bins = [0, 200000, 400000, np.inf]
labels = ['low', 'medium', 'high']
df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)

# Convert categories to numerical values
df['price_category'] = df['price_category'].cat.codes

# Define features and target variable
features = ['size', 'bedrooms', 'age']
X = df[features]
y = df['price_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Select new data for prediction from the original dataset (example: first 5 rows)
new_data = df[features].head(5)

# Predicting the target for new data
new_predictions = model.predict(new_data)
new_predictions_proba = model.predict_proba(new_data)

print("New Data Predictions (0 = low, 1 = medium, 2 = high):", new_predictions)
print("New Data Prediction Probabilities:\n", new_predictions_proba)
