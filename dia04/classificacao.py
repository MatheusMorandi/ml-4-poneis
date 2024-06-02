# %% 

import pandas as pd

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")

# %%

df ["aprovado"] = df["nota"] >= 5

df

# %%

from sklearn import linear_model

rl = linear_model.LogisticRegression(penalty=None, fit_intercept=True)

features = ["cerveja"]

target = "aprovado"

rl.fit(df[features], df[target])

rl_pred = rl.predict(df[features])

# %%

from sklearn import metrics

rl_acc = metrics.accuracy_score(df[target], rl_pred)

rl_acc

# %%

rl_conf = metrics.confusion_matrix(df[target], rl_pred)

rl_conf = pd.DataFrame(rl_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])

rl_conf

# %%
