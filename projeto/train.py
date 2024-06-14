# %%

import pandas as pd

df = pd.read_csv("../data/dados_pontos.csv", sep = ";")

# %%

from sklearn import model_selection

features = df.columns[3:-1]

target = "flActive"

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], df[target], test_size=0.2, random_state=42)

print("Tx Resposta Treino: ", y_train.mean())

print("Tx Resposta Teste: ", y_test.mean())

# %%

