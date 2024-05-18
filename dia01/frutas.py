# %%

import pandas as pd

dados = pd.read_excel("../data/dados_frutas.xlsx")

dados
# %%

filtro_arredondada = dados["Arredondada"] == 1

filtro_suculenta = dados["Suculenta"] == 1

filtro_vermelha = dados["Vermelha"] == 1

filtro_doce = dados["Doce"] == 1

dados[filtro_arredondada & filtro_suculenta & filtro_vermelha & filtro_doce]

# %%

from sklearn import tree

features = ["Arredondada", "Suculenta", "Vermelha", "Doce"]

target = "Fruta"

x = dados[features]

y = dados[target]

# %%


arvore = tree.DecisionTreeClassifier()

arvore.fit(x, y)

# %%

tree.plot_tree(arvore,
                class_names=arvore.classes_,
                feature_names=features,
                filled=True)

# %%
