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


arvore = tree.DecisionTreeClassifier(random_state=42)

arvore.fit(x, y)

# %%

tree.plot_tree(arvore,
                class_names=arvore.classes_,
                feature_names=features,
                filled=True)

# %%

arvore.predict([[1,1,1,1]])


# %%

probabilidade = arvore.predict_proba([[1,1,1,1]])[0]

pd.Series(probabilidade, index=arvore.classes_)

# %%
