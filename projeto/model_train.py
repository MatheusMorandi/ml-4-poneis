# %%

import pandas as pd

from sklearn import model_selection

from sklearn import metrics 

from sklearn import tree 

from sklearn import linear_model

from sklearn import naive_bayes

from sklearn import pipeline

from sklearn import ensemble

from feature_engine import imputation

import scikitplot as skplt

# %%

df = pd.read_csv("../data/dados_pontos.csv", sep=";")

# %%

features = df.columns.tolist()[3:-1]

target = "flActive"

# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], df[target], test_size=0.2, stratify=df[target], random_state=42)

# %%

X_train.isna().sum()

max_avgRecorrencia = X_train["avgRecorrencia"].max()

# %%

imputacao_max = imputation.ArbitraryNumberImputer(variables=["avgRecorrencia"], arbitrary_number=max_avgRecorrencia)

model = ensemble.RandomForestClassifier(random_state=42)

params = {
    "n_estimators": [100,150,250,500],
    "min_samples_leaf": [10,20,30,50,100],
}

grid = model_selection.GridSearchCV(model,
                                    param_grid=params,
                                    n_jobs=-1,
                                    scoring='roc_auc')


meu_pipeline = pipeline.Pipeline([
                                ("imputacao_max", imputacao_max),
                                ("model", grid),
                                ])

meu_pipeline.fit(X_train, y_train)

 # %%

# Predições do Treino

y_train_pred = meu_pipeline.predict(X_train)

y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

# Predições do Teste

y_test_pred = meu_pipeline.predict(X_test)

y_test_proba = meu_pipeline.predict_proba(X_test)

# %%

# Acurácia do Treino

train_acc = metrics.accuracy_score(y_train, y_train_pred)

print("Acurácia da base de treino: ", train_acc)

# Acurácia do Teste

test_acc = metrics.accuracy_score(y_test, y_test_pred)

print("Acurácia da base de teste: ", test_acc)

# Curva ROC do Treino

train_auc = metrics.roc_auc_score(y_train, y_train_proba)

print("Curva ROC da base de treino: ", train_auc)

# Cuva ROC do Teste

test_auc = metrics.roc_auc_score(y_test, y_test_proba[:,1])

print("Curva ROC da base de teste: ", test_auc)

# %%

f_importance = meu_pipeline[-1].best_estimator_.feature_importances_

pd.Series(f_importance, index=features).sort_values(ascending=False)

# %%

skplt.metrics.plot_roc(y_test, y_test_proba)

# %%

skplt.metrics.plot_cumulative_gain(y_test, y_test_proba)

# %%

usuarios_test = pd.DataFrame(
    {"verdadeiro": y_test,
     "proba": y_test_proba[:,1]}
)

usuarios_test = usuarios_test.sort_values("proba", ascending=False)
usuarios_test["sum_verdadeiro"] = usuarios_test["verdadeiro"].cumsum()
usuarios_test["tx captura"]=usuarios_test["sum_verdadeiro"] / usuarios_test["verdadeiro"].sum()
usuarios_test

# %%

skplt.metrics.plot_lift_curve(y_test, y_test_proba)

# %%

usuarios_test.head(100)['verdadeiro'].mean() / usuarios_test['verdadeiro'].mean()

# %%
