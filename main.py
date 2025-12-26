import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    confusion_matrix, fbeta_score, make_scorer, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def boxplot_compare(models_scores, metric):
    """
    Vizuelno poređenje performansi više modela putem boxplot i stripplot grafika.

    Parametri
    ---------
    models_scores : dict
        Rečnik u kojem je ključ naziv modela (str), dok je vrednost lista numeričkih
        rezultata (npr. vrednosti accuracy, F1, F2...) dobijenih iz cross-validacije.

    metric : str
        Naziv metrike koja se poredi (npr. "accuracy", "F2", "F1"). Koristi se za
        generisanje imena kolona i naslova grafika.

    Povratna vrednost
    -----------------
    None
        Funkcija direktno prikazuje poređenje performansi modela kroz boxplot i stripplot,
        bez vraćanja izlazne vrednosti. Boxplot prikazuje raspodelu performansi svakog
        modela, dok stripplot dodaje pojedinačne tačke radi boljeg uvida u varijansu i
        stabilnost modela.
    """
    results = []
    for model_name, scores in models_scores.items():
        for s in scores:
            results.append({"Model": model_name, metric: s})

    df_results = pd.DataFrame(results)

    plt.figure(figsize=(8, 6))
    sns.boxplot(y="Model", x=metric, data=df_results, palette="Set2", hue="Model", legend=False)
    sns.stripplot(y="Model", x=metric, data=df_results, color="black", alpha=0.6)

    plt.title("Model "+metric+" Comparison", fontsize=14)
    plt.ylabel("Model", fontsize=12)
    plt.xlabel("Cross-Validation "+metric, fontsize=12)
    plt.xticks(rotation=15)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluacija klasifikacionog modela nad test skupom podataka.

    Parametri
    ---------
    model : sklearn estimator
        Klasifikacioni model koji se trenira i evaluira.

    X_train : pd.DataFrame ili np.ndarray
        Skup atributa korišćen za treniranje modela.

    X_test : pd.DataFrame ili np.ndarray
        Skup atributa korišćen za testiranje modela.

    y_train : pd.Series ili np.ndarray
        Prave ciljne vrednosti za treniranje modela.

    y_test : pd.Series ili np.ndarray
        Prave ciljne vrednosti za evaluaciju modela.

    Povratna vrednost
    -----------------
    None
        Funkcija ispisuje osnovne klasifikacione metrike (tačnost, preciznost,
        odziv, F1 i F2) i prikazuje matricu konfuzije za dati model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Tacnost je:", accuracy_score(y_test, y_pred))
    print("Preciznost je:", precision_score(y_test, y_pred))
    print("Odziv je:", recall_score(y_test, y_pred))
    print("F2 je:", fbeta_score(y_test, y_pred, beta=2))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.suptitle("Conf. Mat. " + model.__class__.__name__)


def knn_impute(data_set, target, predictor):
    """
    KNN imputacija za izabrani feature

    Parametri
    ---------
    data_set : pd.DataFrame
        Originalni skup podataka sa nedostajućim vrednostima.

    target : str
        Ime kolone za koju se radi imputacija.

    predictor : list of str
        Lista kolona koje se koriste kao prediktori pri KNN imputaciji.

    Povratna vrednost
    -----------------
    np.ndarray
        Imputirane vrednosti za target kolonu u originalnim kategorijama.
    """
    dataframe = data_set.copy()
    encoders = {}

    for cols in [target] + predictor:
        le = LabelEncoder()
        dataframe[cols] = le.fit_transform(dataframe[cols].astype(str))
        encoders[cols] = le

    # imputation frame
    impute_frame = dataframe[[target] + predictor]

    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(impute_frame)

    dataframe[target] = imputed[:, 0].round().astype(int)

    return encoders[target].inverse_transform(dataframe[target])


dataset = pd.read_csv("MushroomDataset/secondary_data.csv", sep=';')

print(dataset.head())
'''
habitat_map = {
    "d": "woods",
    "g": "grasses",
    "l": "leaves",
    "m": "meadows",
    "h": "heaths",
    "p": "paths",
    "w": "waste",
    "u": "urban"
}

counts = dataset['habitat'].value_counts().sort_index()
plt.figure(figsize=(15, 8))
bars = plt.bar(counts.index, counts.values, color="orange")
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.xlabel("Habitat")
plt.ylabel("Count")
plt.title("Distribution by Habitat")

legend_patches = [
    mpatches.Patch(color="orange", label=f"{code} - {desc}")
    for code, desc in habitat_map.items()
]

plt.legend(handles=legend_patches, title="Habitat Categories")

plt.show()

sns.histplot(dataset["cap-diameter"], bins=30)
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.xlabel("Cap Diameter [cm]")
plt.ylabel("Frequency")
plt.title("Histogram of Cap Diameter")
plt.show()

sns.histplot(dataset["stem-width"], bins=30)
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.xlabel("Stem width [mm]")
plt.ylabel("Frequency")
plt.title("Histogram of Stem height")
plt.show()

sns.histplot(dataset["stem-height"], bins=30)
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.xlabel("Stem height [cm]")
plt.ylabel("Frequency")
plt.title("Histogram of Stem height")
plt.show()


print(dataset.describe().T)

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

print(dataset['class'].value_counts(normalize=True))

sns.countplot(x="class", data=dataset)
plt.title("Class Distribution")
plt.show()
'''

'''
Brisanje kolona koje imaju prevelik broj nedostajucih vrednosti (>60%)
'''
for i in dataset.columns:
    if dataset[i].dtypes == 'object':
        if dataset[i].isna().mean() >= 0.6:
            dataset.drop([i], axis=1, inplace=True)


dataset = dataset[dataset['cap-diameter'] <= 30]
dataset = dataset[dataset['stem-width'] <= 60]


dataset['stem-width'] = dataset['stem-width'] / 10


# KNN imputacija feature-a ring-type
df = dataset.copy()
df.drop(['class', 'gill-spacing', 'gill-attachment', 'cap-surface', 'ring-type'], axis=1, inplace=True)
tar_pred_map = {
    "ring-type": ["has-ring", "stem-height", "cap-shape", "gill-attachment"],
    "gill-spacing": df.columns.tolist()
}
for targets, predictors in tar_pred_map.items():
    dataset[targets] = knn_impute(dataset, targets, predictors)

# mode imputacija
for i in dataset.columns:
    if dataset[i].isna().sum() != 0:
        s = dataset[i].value_counts(normalize=True)
        missing = dataset[i].isna()
        dataset.loc[missing, i] = np.random.choice(s.index, p=s.values, size=missing.sum())

'''Enkodovanje za modele tipa stabla'''
data_tree = dataset.copy()
data_lin = dataset.copy()
Encoder = LabelEncoder()
for i in data_tree.columns:
    if data_tree[i].dtype == 'object':
        data_tree[i] = Encoder.fit_transform(data_tree[i])

scaler = StandardScaler()

'''
numerical_columns = []
for i in data_tree.columns:
    if data_tree[i].dtype != 'object':
        numerical_columns.append(i)


correlation_matrix = data_tree[numerical_columns].corr()

plt.figure(1, figsize=(10, 10))
plt.figure(1).suptitle("Matrica korelacija za numericke podatke")
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidth=.5)
plt.show()
'''

y_tree = data_tree['class']
X_tree = data_tree.drop(['class', 'cap-diameter'], axis=1)

y_lin = data_tree['class']
X_lin = pd.get_dummies(data_lin.drop(['class', 'cap-diameter'], axis=1), drop_first=True)
X_lin_scaled = scaler.fit_transform(X_lin)

X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin_scaled, y_lin, test_size=0.3, stratify=y_lin, random_state=42)
X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(X_tree, y_tree, test_size=0.3, stratify=y_tree, random_state=42)

tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
randForest = RandomForestClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
svc = SVC(random_state=42)
lr = LogisticRegression(random_state=42)
KNN = KNeighborsClassifier()

models = [tree, randForest, svc, lr, KNN]
accuracy_values = []    # kolekcija listi rezultata tacnosti cross validation-a
f2_values = []      # kolekcija listi rezultata F2 score-a cross validation-a
f2_scorer = make_scorer(fbeta_score, beta=2)
scorer = {
    "accuracy": "accuracy",
    "f2": f2_scorer
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=True)

for mod in models:
    if isinstance(mod, (DecisionTreeClassifier, RandomForestClassifier)):
        score = cross_validate(mod, X_tree_train, y_tree_train, cv=cv, scoring=scorer, return_train_score=False)
        accuracy_values.append(score['test_accuracy'])
        f2_values.append(score['test_f2'])

        print("--------------- ", mod.__class__.__name__, "---------------")
        print("CV accuracy scores:", score['test_accuracy'])
        print("Mean CV accuracy score:", score['test_accuracy'].mean())
        evaluate_model(mod, X_tree_train, X_tree_test, y_tree_train, y_tree_test)
        sel = SelectFromModel(mod)
        print("Broj koriscenih parametara: ", len(sel.get_support()[sel.get_support() == True]))

        # Podesavanje hiperparametara Random Foresta
        if isinstance(mod, RandomForestClassifier):
            hyperparams = [
                {
                    'n_estimators': [200, 500],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ]
            grid = GridSearchCV(mod, hyperparams, cv=cv, scoring=scorer, refit="f2")
            grid.fit(X_tree_train, y_tree_train)
            print("najbolja kombinacija je: ", grid.best_params_)

            mod.n_estimators = grid.best_params_['n_estimators']
            mod.max_depth = grid.best_params_['max_depth']
            mod.min_samples_split = grid.best_params_['min_samples_split']
            mod.min_samples_leaf = grid.best_params_['min_samples_leaf']

            print("Metrike ", mod.__class__.__name__, "nakon podesavanja hiperparametara:")
            evaluate_model(mod, X_tree_train, X_tree_test, y_tree_train, y_tree_test)
        else:   # podesavanje hiperparametara Decision Tree-a
            hyperparams = [
                {
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ]
            grid = GridSearchCV(mod, hyperparams, cv=cv, scoring=scorer, refit="f2")
            grid.fit(X_tree_train, y_tree_train)
            print("najbolja kombinacija je: ", grid.best_params_)

            mod.max_depth = grid.best_params_['max_depth']
            mod.min_samples_split = grid.best_params_['min_samples_split']
            mod.min_samples_leaf = grid.best_params_['min_samples_leaf']

            print("Metrike ", mod.__class__.__name__, "nakon podesavanja hiperparametara:")
            evaluate_model(mod, X_tree_train, X_tree_test, y_tree_train, y_tree_test)
    else:
        score = cross_validate(mod, X_lin_train, y_lin_train, cv=cv, scoring=scorer, return_train_score=False)
        accuracy_values.append(score['test_accuracy'])
        f2_values.append(score['test_f2'])

        print("--------------- ", mod.__class__.__name__, "---------------")
        print("CV accuracy scores:", score['test_accuracy'])
        print("Mean CV accuracy score:", score['test_accuracy'].mean())
        evaluate_model(mod, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

        if isinstance(mod, SVC):    # podesavanje hiperparametara SVC-a
            hyperparams = [
                {
                    'gamma': ['scale', 'auto'],
                    'C': [1, 5, 10]
                }
            ]
            grid = GridSearchCV(mod, hyperparams, cv=cv, scoring=scorer, refit="f2")
            grid.fit(X_lin_train, y_lin_train)
            print("Najbolja kombinacija je: ", grid.best_params_)
            mod.gamma = grid.best_params_['gamma']
            mod.C = grid.best_params_['C']

            print("Metrike ", mod.__class__.__name__, "nakon podesavanja hiperparametara:")
            evaluate_model(mod, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

        elif isinstance(mod, KNeighborsClassifier):     # podesavanje hiperparametara KNN-a
            hyperparams = [
                {
                    'n_neighbors': [5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            ]
            grid = GridSearchCV(mod, hyperparams, cv=cv, scoring=scorer, refit="f2")
            grid.fit(X_lin_train, y_lin_train)
            print("Najbolja kombinacija parametara: ", grid.best_params_)

            mod.n_neighbors = grid.best_params_['n_neighbors']
            mod.weights = grid.best_params_['weights']
            mod.metric = grid.best_params_['metric']

            print("Metrike ", mod.__class__.__name__, "nakon podesavanja hiperparametara:")
            evaluate_model(mod, X_lin_train, X_lin_test, y_lin_train, y_lin_test)

        else:   # podesavanje hiperparametara Logistic Regression-a
            hyperparams = [
                {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            ]
            grid = GridSearchCV(mod, hyperparams, cv=cv, scoring=scorer, refit="f2")
            grid.fit(X_lin_train, y_lin_train)
            print("Najbolja kombinacija parametara: ", grid.best_params_)

            mod.C = grid.best_params_['C']
            mod.penalty = grid.best_params_['penalty']
            mod.solver = grid.best_params_['solver']

            print("Metrike ", mod.__class__.__name__, "nakon podesavanja hiperparametara:")
            evaluate_model(mod, X_lin_train, X_lin_test, y_lin_train, y_lin_test)
    print()

models_accuracy = {
    "Decision Tree": accuracy_values[0],
    "Random Forest": accuracy_values[1],
    "SVC": accuracy_values[2],
    "Logistic Regression": accuracy_values[3],
    "KNN": accuracy_values[4]
}

models_f2 = {
    "Decision Tree": f2_values[0],
    "Random Forest": f2_values[1],
    "SVC": f2_values[2],
    "Logistic Regression": f2_values[3],
    "KNN": f2_values[4]
}

boxplot_compare(models_accuracy, "Accuracy")
boxplot_compare(models_f2, "F2 score")

plt.figure(figsize=(10, 8))

for mod in models:

    # Različiti ulazi za dva skupa osobina
    if isinstance(mod, (DecisionTreeClassifier, RandomForestClassifier)):
        X_test_used = X_tree_test
        y_test_used = y_tree_test
    else:
        X_test_used = X_lin_test
        y_test_used = y_lin_test

    # Dobijanje score vrednosti
    if hasattr(mod, "predict_proba"):
        try:
            y_scores = mod.predict_proba(X_test_used)[:, 1]
        except:
            # Npr. ako model ima predict_proba ali ga ne podržava u nekom modu
            y_scores = mod.decision_function(X_test_used)
    else:
        # Ovo važi za SVC bez probability=True
        y_scores = mod.decision_function(X_test_used)

    # ROC podaci
    fpr, tpr, _ = roc_curve(y_test_used, y_scores)
    auc_score = auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, linewidth=2, label=f"{mod.__class__.__name__} (AUC = {auc_score:.3f})")

# Dijagonala
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")

plt.title("ROC Curves for All Models", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate (Recall)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
