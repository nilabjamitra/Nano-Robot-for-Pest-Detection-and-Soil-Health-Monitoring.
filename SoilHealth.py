import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load the provided dataset
df = pd.read_csv('soil data.csv')

# Drop irrelevant and unnamed columns
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col or col == "ID"], errors='ignore')

# Clean up column names by stripping extra spaces
df.columns = df.columns.str.strip()

# Select relevant features for soil health analysis
relevant_columns = ['pH', 'N_NO3 ppm', 'P ppm', 'K ppm', 'EC mS/cm', 'O.M. %']
df = df[relevant_columns].dropna()

# Adding a target column for Soil Health (1: Healthy, 0: Unhealthy)
# Defining healthy soil as having balanced pH, nutrients, and sufficient organic matter
def classify_soil(row):
    if 6.0 <= row['pH'] <= 7.5 and row['N_NO3 ppm'] > 10 and row['P ppm'] > 15 and row['K ppm'] > 150 and row['O.M. %'] > 1.0:
        return 1
    return 0

df['Soil_Health'] = df.apply(classify_soil, axis=1)

# Splitting data into features and target
X = df[['pH', 'N_NO3 ppm', 'P ppm', 'K ppm', 'EC mS/cm', 'O.M. %']]
y = df['Soil_Health']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Feature importance visualization
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Accuracy plot
accuracy = accuracy_score(y_test, y_pred)
plt.bar(['Accuracy'], [accuracy], color='teal')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Score')
plt.show()

# Function to provide soil health suggestions
def soil_health_suggestions(nanorobot_data):
    # Print input data
    print("Input Data:", nanorobot_data)

    # Predict soil health
    prediction = model.predict([nanorobot_data])[0]

    # Print prediction
    print("Prediction:", prediction)

    if prediction == 1:
        return "Soil is healthy. Maintain current practices."
    else:
        suggestions = []
        if nanorobot_data[0] < 6.0:
            suggestions.append("Increase soil pH by adding lime.")
        elif nanorobot_data[0] > 7.5:
            suggestions.append("Reduce soil pH by adding sulfur or organic matter.")
        if nanorobot_data[1] < 10:
            suggestions.append("Increase nitrogen levels using organic fertilizers or compost.")
        if nanorobot_data[2] < 15:
            suggestions.append("Add phosphorus using rock phosphate or bone meal.")
        if nanorobot_data[3] < 150:
            suggestions.append("Increase potassium levels using potash or wood ash.")
        if nanorobot_data[5] < 1.0:
            suggestions.append("Enhance organic matter using compost or green manure.")

        # Print suggestions before returning
        print("Suggestions:", suggestions)

        return f"Soil is unhealthy. Suggestions to improve soil health:\n{chr(10).join(suggestions)}"

# Example input from nanorobots (pH, Nitrogen, Phosphorus, Potassium, EC, Organic Matter)
nanorobot_data = [6.5, 12, 20, 180, 0.3, 1.2]  # Replace with real data

# Get suggestions
result = soil_health_suggestions(nanorobot_data)
print("\nSoil Health Analysis Result:")
print(result)
