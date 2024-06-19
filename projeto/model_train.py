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

model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)

meu_pipeline = pipeline.Pipeline([
                                ("imputacao_max", imputacao_max),
                                ("model", model),
                                ])

meu_pipeline.fit(X_train, y_train)

# %%

# Predições do Treino

y_train_pred = meu_pipeline.predict(X_train)

y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

# Predições do Teste

y_test_pred = meu_pipeline.predict(X_test)

y_test_proba = meu_pipeline.predict_proba(X_test)[:,1]

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

test_auc = metrics.roc_auc_score(y_test, y_test_proba)

print("Curva ROC da base de teste: ", test_auc)




# %%