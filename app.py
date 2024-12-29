from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)


path = 'C://Users//world//OneDrive//Desktop//INTERNSHIPS//ORINSON//asteroid_dataset.csv'
data = pd.read_csv(path, low_memory=False)
threshold = 0.85 * len(data)
data = data.dropna(thresh=threshold, axis=1)
data = data.dropna()

data.columns = data.columns.str.strip()
target = 'pha'

X = data[['H', 'e', 'a', 'q', 'i']]
y = data[target]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_pred_proba_log = log_reg.predict_proba(X_test)[:, 1]

dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_tree = dec_tree.predict(X_test)
y_pred_proba_tree = dec_tree.predict_proba(X_test)[:, 1]

rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(X_train, y_train)
y_pred_forest = rand_forest.predict(X_test)
y_pred_proba_forest = rand_forest.predict_proba(X_test)[:, 1]


def evaluate_model(y_test, y_pred, y_pred_proba=None):
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba) if y_pred_proba is not None else None
    precision = precision_score(y_test, y_pred, pos_label='Y')
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    return accuracy, logloss, precision, roc_auc


metrics = {
    "Logistic Regression": evaluate_model(y_test, y_pred_log, y_pred_proba_log),
    "Decision Tree": evaluate_model(y_test, y_pred_tree, y_pred_proba_tree),
    "Random Forest": evaluate_model(y_test, y_pred_forest, y_pred_proba_forest),
}

conf_matrices = {
    "Logistic Regression": confusion_matrix(y_test, y_pred_log),
    "Decision Tree": confusion_matrix(y_test, y_pred_tree),
    "Random Forest": confusion_matrix(y_test, y_pred_forest),
}

roc_curves = {
    "Logistic Regression": roc_curve(y_test, y_pred_proba_log, pos_label='Y'),
    "Decision Tree": roc_curve(y_test, y_pred_proba_tree, pos_label='Y'),
    "Random Forest": roc_curve(y_test, y_pred_proba_forest, pos_label='Y'),
}


def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Routes
@app.route('/')
def home():
    return render_template('index.html', models=list(metrics.keys()))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    selected_model = request.form['model']
    acc, logloss, precision, roc_auc = metrics[selected_model]
    cm = conf_matrices[selected_model]

    
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {selected_model}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_image = plot_to_base64(fig_cm)
    plt.close(fig_cm)

    
    if selected_model == "Random Forest":
        selected_features = ['H', 'e', 'a', 'q', 'i']
        importances = rand_forest.feature_importances_
        fig_importance = plt.figure(figsize=(8, 5))
        plt.bar(selected_features, importances, color='teal')
        plt.title("Feature Importance: Random Forest")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        importance_image = plot_to_base64(fig_importance)
        plt.close(fig_importance)
    else:
        importance_image = None

    return render_template(
        'evaluate.html',
        model=selected_model,
        accuracy=acc,
        logloss=logloss,
        precision=precision,
        roc_auc=roc_auc,
        cm_image=cm_image,
        importance_image=importance_image,
    )

@app.route('/roc_curve')
def roc_curve_view():
    fig_roc = plt.figure(figsize=(8, 6))
    for model, (fpr, tpr, _) in roc_curves.items():
        auc = metrics[model][3]
        plt.plot(fpr, tpr, label=f"{model} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    roc_image = plot_to_base64(fig_roc)
    plt.close(fig_roc)
    return render_template('roc_curve.html', roc_image=roc_image)


if __name__ == '__main__':
    app.run(debug=True)
