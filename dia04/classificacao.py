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

rl_precision = metrics.precision_score(df[target], rl_pred)

print("Precisão da Regressão Logística: ", rl_precision)

rl_recall = metrics.recall_score(df[target], rl_pred)

print("Recall da Regressão Logística: ", rl_recall)

rl_conf = metrics.confusion_matrix(df[target], rl_pred)

rl_conf = pd.DataFrame(rl_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

print("Matriz de Confusão da Regressão Logística: \n", rl_conf)

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

arvore.fit(df[features], df[target])

arvore_pred = arvore.predict(df[features])

arvore_pred

arvore_acc = metrics.accuracy_score(df[target], arvore_pred)

print("Acurácia da Árvore: ", arvore_acc)

arvore_precision = metrics.precision_score(df[target], arvore_pred)

print("Precisão da Árvore: ", arvore_precision)

arvore_recall = metrics.recall_score(df[target], arvore_pred)

print("Recall da Árvore: ", arvore_recall)

arvore_conf = metrics.confusion_matrix(df[target], arvore_pred)

arvore_conf = pd.DataFrame(arvore_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

print("Matriz de Confusão da Árvore: \n", arvore_conf)

# %%

from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

nb.fit(df[features], df[target])

nb_pred = nb.predict(df[features])

nb_pred

nb_acc = metrics.accuracy_score(df[target], nb_pred)

print("Acurácia Naive Bayes: ", nb_acc)

nb_precision = metrics.precision_score(df[target], nb_pred)

print("Precisão Naives Bayes: ", nb_precision)

nb_recall = metrics.recall_score(df[target], nb_pred)

print("Recall Naives Bayes: ", nb_recall)

nb_conf = metrics.confusion_matrix(df[target], nb_pred)

nb_conf = pd.DataFrame(nb_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

print("Matriz de Confusão do Naives Bayes: \n", nb_conf)

# %%

nb_proba = nb.predict_proba(df[features])[:,1]

nb_pred = nb_proba > 0.7

nb_acc = metrics.accuracy_score(df[target], nb_pred)

print("Acurácia Naive Bayes:", nb_acc)

nb_precision = metrics.precision_score(df[target], nb_pred)

print("Precisão Naive Bayes:", nb_precision)

nb_recall = metrics.recall_score(df[target], nb_pred)

print("Recall Naive Bayes:", nb_recall)

# %%

import matplotlib.pyplot as plt

roc_curve = metrics.roc_curve(df[target], nb_proba)

plt.plot(roc_curve[0], roc_curve[1] )

plt.grid(True)

plt.plot([0,1], [0,1], '--')

plt.show()

# %%

roc_auc = metrics.roc_auc_score(df[target], nb_proba)

roc_auc

# %%

