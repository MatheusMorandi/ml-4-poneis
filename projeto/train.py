# %%

import pandas as pd

df = pd.read_csv("../data/dados_pontos.csv", sep = ";")

# %%

from sklearn import model_selection

features = df.columns[3:-1]

target = "flActive"

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], df[target], test_size=0.2, random_state=42, stratify= df[target])

print("Tx Resposta Treino: ", y_train.mean())

print("Tx Resposta Teste: ", y_test.mean())

# %%

input_avgRecorrencia = X_train["avgRecorrencia"].max()

X_train["avgRecorrencia"] = X_train["avgRecorrencia"].fillna(input_avgRecorrencia)

X_test["avgRecorrencia"] = X_test["avgRecorrencia"].fillna(input_avgRecorrencia)

# %%

from sklearn import metrics

from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=6,
                                    min_samples_leaf=50,
                                    random_state=42)

arvore.fit(X_train, y_train)

# Métricas do treino

arvore_pred_train = arvore.predict(X_train)

arvore_acc_train = metrics.accuracy_score(y_train, arvore_pred_train)

print("Acurácia da Árvore(treino): ", arvore_acc_train)

# Métricas do teste

arvore_pred_test = arvore.predict(X_test)

arvore_acc_test = metrics.accuracy_score(y_test, arvore_pred_test)

print("Acurácia da Árvore(teste): ", arvore_acc_test)

# Métricas (proba) do treino

arvore_proba_train = arvore.predict_proba(X_train)[:,1]

arvore_auc_train = metrics.roc_auc_score(y_train, arvore_proba_train)

print("AUC da Árvore(treino): ", arvore_auc_train)

# Métricas (proba) do teste

arvore_proba_test = arvore.predict_proba(X_test)[:,1]

arvore_auc_test = metrics.roc_auc_score(y_test, arvore_proba_test)

print("AUC da Árvore(teste): ", arvore_auc_test)

# %%
