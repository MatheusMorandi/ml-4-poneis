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

from sklearn import tree 

arvore = tree.DecisionTreeRegressor(max_depth=2)

arvore.fit(df[["cerveja"]], df["nota"])

# %%

y_estimado_arvore = arvore.predict(x)

plt.plot(df["cerveja"], df["nota"], "o")

plt.plot(x, y_estimado, "-")

plt.plot(x, y_estimado_arvore, "-")

plt.grid(True)

plt.title("Relação entre Nota e quantidade de Cervejas")

plt.ylim(0, 11)

plt.xlim(0, 11)

plt.ylabel("Nota")

plt.xlabel("Cerveja")

plt.legend(["Pontos", "Regressão Linear", "Árvore de Decisão"])

plt.show()


# %%
