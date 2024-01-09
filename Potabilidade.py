# Autor: Gabriel Barreto
# baseado no codigo do professor DANIEL CAVALCANTI JERONYMO
# fonte do dataset: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import tree, metrics
import pydotplus
import os

def sortSecond(val):
    return val[1]

set_treino = pd.read_csv('water_potability.csv')

# O dado de saida Potability ja esta em valores de 0 (nao potavel) e 1 (potavel)
# As caracteristicas de cada amostra analisada serao os parametros para indicar a potabilidade da agua

todas_saidas = set_treino['Potability'].values
tamanhodoteste = len(todas_saidas) - 5

y = todas_saidas[0:tamanhodoteste]
y_testefinal = todas_saidas[tamanhodoteste:len(todas_saidas)]

# Usaremos todas as caracteristicas menos a de potabilidade(saida)
colunas = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
           'Organic_carbon', 'Trihalomethanes', 'Turbidity']

valores = set_treino[list(colunas)].values

# Como algumas celulas estao em branco, vamos preenche-las com os valores mais frequentes de cada coluna
# Percebi que ao substituir a media dos valores, a taxa de acerto caia pela metade, enquanto ao utilizar os valores
# mais frequentes fazia a taxa de acerto aumentar
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
todos_valores = imp.fit_transform(valores)

# Treinando o modelo com todas as amostras menos as 5 ultimas, que serao usadas como teste
X = todos_valores[0:tamanhodoteste]
# Amostras de teste
X_testefinal = todos_valores[tamanhodoteste:len(todos_valores)]

# A profundidade da arvore nao foi ajustada, pois em todas tentativas o resultado era impreciso
# A arvore gerada, no entanto, fica muito grande, mas tem uma chance de acerto maior
# O criterio Gini de impureza gera decisoes mais acertivas que o por entropia
clf = tree.DecisionTreeClassifier(criterion='gini')#, max_depth=5)
clf = clf.fit(X, y)

# Print das importancias de cada parametro da arvore de decisao
importancia = clf.feature_importances_
importancias = [(colunas[i], importancia[i]) for i in range(len(colunas))]
importancias.sort(reverse=True, key=sortSecond)
print(importancias)
# Perceba que todas caracteristicas tem relativamente a mesma importancia +-10%,
# por isso decidi nao tirar nenhuma


print('Precisão do modelo:', metrics.accuracy_score(y_testefinal, clf.predict(X_testefinal)))
# Testando algumas entradas para verificar a potabilidade
for i in range(len(X_testefinal)):
    print('Potabilidade Predita: ', clf.predict([X_testefinal[i]]), ' Potabilidade Real: ', y_testefinal[i])

# Cria a imagem da arvore de decisao
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=colunas)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('potabilidade.png')

