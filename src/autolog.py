import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='ayushmansingh.199978', repo_name='MLOPS-Experiments-with-MLFlow', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/ayushmansingh.199978/MLOPS-Experiments-with-MLFlow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 15

mlflow.autolog()
mlflow.set_experiment("MLOPS-Expr-1") # Set the experiment name

with mlflow.start_run(): # Experiment can be set from here as well (experiment_id=id_no.)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric("accuracy", accuracy)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)

    #Creating confusion matrix
    cm = confusion_matrix(y_test, y_pred) # Compute the confusion matrix
    plt.figure(figsize=(8, 6)) # Set the figure size for the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names) # Confusion matrix
    plt.xlabel('Predicted') # Label for the predicted values
    plt.ylabel('Actual') # Label for the actual values
    plt.title('Confusion Matrix') # Title for the confusion matrix
    plt.savefig("confusion_matrix.png") # Save the confusion matrix as an image
    # mlflow.log_artifact("confusion_matrix.png") # Log the confusion matrix as an artifact
    # mlflow.log_artifact(__file__) # Log the current script as an artifact
    mlflow.set_tags({"Author": "Ayushmaan", "Project": "Wine Classification"})
    #mlflow.sklearn.log_model(sk_model=rf, name="random-forest-model", skops_trusted_types=["sklearn.tree._tree.Tree"])
    print(f"Accuracy: {accuracy}")