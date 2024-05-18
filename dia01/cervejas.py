# %%

import pandas as pd

dados = pd.read_excel("../data/dados_cerveja.xlsx")

dados

# %%

features = ["temperatura", "copo", "espuma", "cor"]

target = "classe"

x = dados[features]

y = dados[target]

# %%

x = x.replace({

    "mud": 1, "pint": 0,
    "sim": 1, "não": 0,
    "escura": 1, "clara": 0,

})


# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier()

arvore.fit(x, y)

# %%

tree.plot_tree(
    arvore,
    class_names=arvore.classes_,
    feature_names=features,
    filled=True
)

# %%

probabilidade = arvore.predict_proba([[-1,1,1,1]])[0]

pd.Series(probabilidade, index=arvore.classes_)


# %%
