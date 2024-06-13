# %% 

import pandas as pd

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")

# %%

from sklearn import linear_model

df ["aprovado"] = df["nota"] >= 5

rl = linear_model.LogisticRegression(penalty=None, fit_intercept=True)

features = ["cerveja"]

target = "aprovado"

rl.fit(df[features], df[target])

rl_pred = rl.predict(df[features])

# %%

from sklearn import metrics

rl_acc = metrics.accuracy_score(df[target], rl_pred)

print("Acurácia da Regressão Logística: ", rl_acc)

rl_conf = metrics.confusion_matrix(df[target], rl_pred)

rl_conf = pd.DataFrame(rl_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

print("Matriz de Confusão da Regressão Logística: \n", rl_conf)

rl_precision = metrics.precision_score(df[target], rl_pred)

print("Precisão da Regressão Logística: ", rl_precision)

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

arvore.fit(df[features], df[target])

arvore_pred = arvore.predict(df[features])

arvore_pred

arvore_acc = metrics.accuracy_score(df[target], arvore_pred)

print("Acurácia da Árvore: ", arvore_acc)

arvore_conf = metrics.confusion_matrix(df[target], arvore_pred)

arvore_conf = pd.DataFrame(arvore_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

print("Matriz de Confusão da Árvore: \n", arvore_conf)

arvore_precision = metrics.precision_score(df[target], arvore_pred)

print("Precisão da Árvore: ", arvore_precision)

# %%

from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

nb.fit(df[features], df[target])

nb_pred = nb.predict(df[features])

nb_pred

nb_acc = metrics.accuracy_score(df[target], nb_pred)

print("Acurácia Naive Bayes: ", nb_acc)

nb_conf = metrics.confusion_matrix(df[target], nb_pred)

nb_conf = pd.DataFrame(nb_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

print("Matriz de Confusão do Naives Bayes: \n", nb_conf)

nb_precision = metrics.precision_score(df[target], nb_pred)

print("Precisão do Naives Bayes: ", nb_precision)

# %%
