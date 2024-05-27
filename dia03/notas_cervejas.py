# %%

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_excel('../data/dados_cerveja_nota.xlsx')

df

# %%

plt.plot(df["cerveja"], df["nota"], "o")

plt.grid(True)

plt.title("Relação entre Nota e quantidade de Cervejas")

plt.ylim(0, 11)

plt.xlim(0, 11)

plt.ylabel("Nota")

plt.xlabel("Cerveja")

plt.show()

# %%

from sklearn import linear_model

rl = linear_model.LinearRegression()

rl.fit(df[["cerveja"]], df["nota"])

# %%

x = df[["cerveja"]].drop_duplicates()

y_estimado = rl.predict(x)


plt.plot(df["cerveja"], df["nota"], "o")

plt.plot(x, y_estimado, "-")

plt.grid(True)

plt.title("Relação entre Nota e quantidade de Cervejas")

plt.ylim(0, 11)

plt.xlim(0, 11)

plt.ylabel("Nota")

plt.xlabel("Cerveja")

plt.show()

# %%
