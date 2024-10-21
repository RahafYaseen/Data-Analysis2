# Diabetes Prediction Project

This project aims to analyze diabetes data and build a prediction model using the Random Forest algorithm.

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Data](#data)
- [Analysis](#analysis)
- [Model](#model)
- [Evaluation](#evaluation)
- [Run the Project](#run-the-project)

## Description

This project uses diabetes data to analyze the relationship between various features (such as age and BMI) and the likelihood of diabetes. A Random Forest algorithm is employed to train a model on the data and predict outcomes.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the requirements using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn

Data
You can download the dataset from this link (https://www.kaggle.com/uciml/pima-indiansdiabetes-database).

Analysis
Data exploration is performed through visualizing the distributions of age and BMI, as well as the relationship between glucose levels and outcomes.

###Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


###BMI Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['BMI'], bins=30, kde=True)
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()


###Glucose Levels by Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outcome', y='Glucose', data=data)
plt.title('Glucose Levels by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Glucose')
plt.show()


###Model
###The Random Forest algorithm is used to build a prediction model.
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestClassifier(n_estimators=100, random_state=125)
rf_model.fit(X_train_scaled, y_train)

###Evaluation
###The model is evaluated using accuracy, confusion matrix, and classification report.
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_rf}")
print("Confusion Matrix:")
print(conf_matrix_rf)
print("Classification Report:")
print(class_report_rf)


###Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix for Random Forest Classifier")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


python your_script.py


### Notes:
- Make sure to update any links or file names as necessary.
- Feel free to add more details or sections based on your project needs.
