# %%

import pandas as pd

df = pd.read_parquet("../data/dados_clones.parquet")

df
# %%

df.groupby(["Status "])[["Estatura(cm)", "Massa(em kilos)"]].mean()

# %%

df["Status_bool"] = df["Status "] == "Apto"

df

# %%

df.groupby(["Distância Ombro a ombro"])["Status_bool"].mean()

# %%

df.groupby(["Tamanho do crânio"])["Status_bool"].mean()

# %%

df.groupby(["Tamanho dos pés"])["Status_bool"].mean()

# %%

df.groupby(["General Jedi encarregado"])["Status_bool"].mean()

# %%

features = [
    "Massa(em kilos)",
    "Estatura(cm)",
    "Distância Ombro a ombro",
    "Tamanho do crânio", 
    "Tamanho dos pés"]

cat_features = [
    "Distância Ombro a ombro",
    "Tamanho do crânio", 
    "Tamanho dos pés"]

x = df[features]

from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables= cat_features)

onehot.fit(x)

x = onehot.transform(x)

x

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

arvore.fit(x, df["Status "])

# %%

tree.plot_tree(arvore,
                class_names= arvore.classes_,
                feature_names= x.columns,
                filled= True)

# %%

