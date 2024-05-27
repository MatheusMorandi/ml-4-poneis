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

