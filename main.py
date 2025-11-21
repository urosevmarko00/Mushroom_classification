import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("MushroomDataset/secondary_data.csv", sep=';')

print(dataset.head())

'''
for i in dataset.columns:
    if dataset[i].dtypes != 'object':
        plt.figure().suptitle(i + " u odnosu na jestivost")
        sns.scatterplot(x='class', y=i, data=dataset)

plt.show()
'''

'''print(dataset.describe().T)

print(dataset.columns)
print()
print(dataset.dtypes)
print()

for i in dataset.columns:
    if dataset[i].dtypes == 'object':
        print(i)
        print(dataset[i].unique())

print()
dataset.info()
print()

print(dataset.isna().sum())
print()
'''

'''
Brisanje kolona koje imaju prevelik broj nedostajucih vrednosti (>60%)
'''
for i in dataset.columns:
    if dataset[i].dtypes == 'object':
        if dataset[i].isna().mean() >= 0.6:
            dataset.drop([i], axis=1, inplace=True)


dataset['stem-width'] = dataset['stem-width'] / 10

for i in dataset.columns:
    if dataset[i].isna().sum() != 0:
        s = dataset[i].value_counts(normalize=True)
        missing = dataset[i].isna()
        dataset.loc[missing, i] = np.random.choice(s.index, p=s.values, size=missing.sum())

'''print()
dataset.info()
print()

print(dataset.isna().sum())
print()'''
'''Enkodovanje za modele tipa stabla'''
data_tree = dataset.copy()
Encoder = LabelEncoder()
for i in data_tree.columns:
    if data_tree[i].dtype == 'object':
        data_tree[i] = Encoder.fit_transform(data_tree[i])


scaler = StandardScaler()

'''numerical_columns = []
for i in dataset.columns:
    if dataset[i].dtype != 'object':
        numerical_columns.append(i)


correlation_matrix = dataset[numerical_columns].corr()

plt.figure(1, figsize=(10, 10))
plt.figure(1).suptitle("Matrica korelacija za numericke podatke")
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidth=.5)
plt.show()
'''

y_tree = data_tree['class']
X_tree = data_tree.drop(['class'], axis=1)

y_lin = data_tree['class']
X_lin = pd.get_dummies(dataset.drop('class', axis=1), drop_first=True)
X_lin_scaled = scaler.fit_transform(X_lin)

X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin_scaled, y_lin, test_size=0.3, stratify=y_lin, random_state=42)
X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X_tree, y_tree, test_size=0.3, stratify=y_tree, random_state=42)

tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
randForest = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
svm = SVC(C=1.0, kernel='rbf', probability=False, random_state=42)
lr = LogisticRegression(max_iter=500, random_state=42)
KNN = KNeighborsClassifier(n_neighbors=5)

models = [tree, randForest, svm, lr, KNN]
scores = []
predictions = []

for mod in models:
    if isinstance(mod, (DecisionTreeClassifier, RandomForestClassifier)):
        mod.fit(X_tree_train, y_tree_train)
        score = cross_val_score(mod, X_tree_train, y_tree_train, cv=5)
        scores.append(score)
        sel = SelectFromModel(mod)
        print("--------------- ", mod.__class__.__name__, "---------------")
        print("CV scores:", score)
        print("Mean CV score:", score.mean())
        y_predict = mod.predict(X_tree_test)
        predictions.append(y_predict)

        print("Broj koriscenih parametara: ", len(sel.get_support()[sel.get_support()==True]))
        print("Tacnost je:", accuracy_score(y_tree_test, y_predict))
        print("Preciznost je:", precision_score(y_tree_test, y_predict))
        print("Odziv je:", recall_score(y_tree_test, y_predict))
        print("F1 je:", f1_score(y_tree_test, y_predict))
        ConfusionMatrixDisplay(confusion_matrix(y_tree_test, y_predict)).plot()
        plt.suptitle("Conf. Mat. " + mod.__class__.__name__)

        # Barplot feature-a po vaznosti i podesavanje hiperparametara Random Foresta
        if isinstance(mod, RandomForestClassifier):
            importances = mod.feature_importances_
            feature_names = X_tree.columns
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            print(feature_importance_df.head())
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=feature_importance_df.head(16),
                x='Importance',
                y='Feature',
                palette='viridis',
                hue='Feature',
                legend=False
            )
            plt.title("Most Important Features (Random Forest)")
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.tight_layout()

            '''hyperparams = [
                {
                    'n_estimators': [100, 300, 700],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ]
            grid = GridSearchCV(mod, hyperparams, cv=4, scoring='accuracy')
            grid.fit(X_tree_train, y_tree_train)
            print("najbolja kombinacija je: ", grid.best_params_)

            mod.n_estimators = grid.best_params_['n_estimators']
            mod.max_depth = grid.best_params_['max_depth']
            mod.min_samples_split = grid.best_params_['min_samples_split']
            mod.min_samples_leaf = grid.best_params_['min_samples_leaf']

            mod.fit(X_tree_train, y_tree_train)
            y_predict = mod.predict(X_tree_test)
            print("Tacnost randForest (posle podesavanja) je: ", accuracy_score(y_tree_test, y_predict))'''
    else:
        mod.fit(X_lin_train, y_lin_train)
        score = cross_val_score(mod, X_lin_train, y_lin_train, cv=5)
        scores.append(score)
        print("--------------- ", mod.__class__.__name__, "---------------")
        print("CV scores:", score)
        print("Mean CV score:", score.mean())
        y_predict = mod.predict(X_lin_test)
        predictions.append(y_predict)

        print("Tacnost je:", accuracy_score(y_lin_test, y_predict))
        print("Preciznost je:", precision_score(y_lin_test, y_predict))
        print("Odziv je:", recall_score(y_lin_test, y_predict))
        print("F1 je:", f1_score(y_lin_test, y_predict))
        ConfusionMatrixDisplay(confusion_matrix(y_lin_test, y_predict)).plot()
        plt.suptitle("Conf. Mat. " + mod.__class__.__name__)

        '''if isinstance(mod, SVC):
            hyperparams = [
                {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                }
            ]

            grid = GridSearchCV(mod, hyperparams, cv=4, scoring='accuracy')
            grid.fit(X_lin_train, y_lin_train)
            print("Najboli kernel za SVC: ", grid.best_params_)
            mod.kernel = grid.best_params_['kernel']
            mod.fit(X_lin_train, y_lin_train)
            y_predict = mod.predict(X_lin_test)
            print("Tacnost SVC (posle podesavanja) je: ", accuracy_score(y_lin_test, y_predict))'''
    print()
results1 = []

models = {
    "Decision Tree": scores[0],
    "Random Forest": scores[1],
    "SVM": scores[2],
    "Logistic Regression": scores[3],
    "KNN": scores[4]
}

for model_name, scores in models.items():
    for s in scores:
        results1.append({"Model": model_name, "Accuracy": s})

df_results1 = pd.DataFrame(results1)
print(df_results1)

plt.figure(figsize=(8, 6))
sns.boxplot(y="Model", x="Accuracy", data=df_results1, palette="Set2", hue="Model", legend=False)
sns.stripplot(y="Model", x="Accuracy", data=df_results1, color="black", alpha=0.6)

plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Cross-Validation Accuracy", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=15)


results2 = []

models2 = {
    "Decision Tree": fbeta_score(y_tree_test, predictions[0], beta=2),
    "Random Forest": fbeta_score(y_tree_test, predictions[1], beta=2),
    "SVM": fbeta_score(y_lin_test, predictions[2], beta=2),
    "Logistic Regression": fbeta_score(y_lin_test, predictions[3], beta=2),
    "KNN": fbeta_score(y_lin_test, predictions[4], beta=2)
}

for model_name2, scores2 in models2.items():
    results2.append({"Model": model_name2, "F2": scores2})

df_results2 = pd.DataFrame(results2)
print(df_results2.head())

plt.figure(figsize=(8, 6))
sns.boxplot(y="Model", x="F2", data=df_results2, palette="Set2", hue="Model", legend=False)
sns.stripplot(y="Model", x="F2", data=df_results2, color="black", alpha=0.6)

plt.title("Model F2 Comparison", fontsize=14)
plt.ylabel("F2 Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=15)
plt.show()
