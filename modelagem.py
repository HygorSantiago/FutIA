import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv('/home/usuario/Área de Trabalho/provisorio/modelo1.csv')

x = df.drop(columns=['Target'])
y = df['Target']

from sklearn.feature_selection import SelectKBest

k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(x,y)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(x.columns, k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda X: X[1])))

k_best_features_final = dict(ordered_pairs[:15])
best_features = k_best_features_final.keys()
print ('')
print ("Melhores features:")
print (k_best_features_final)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

def treino(modelo,x_treino, x_teste, y_treino, y_teste):
    modelo.fit(x_treino, y_treino)
    print(modelo)
    acc = modelo.score(x_teste,y_teste)
    print(acc*100)

def treino_grid(modelo,parametros,x,y):
    grid = GridSearchCV(modelo,parametros,cv=10,n_jobs=-1,verbose=3)
    grid.fit(x,y)
    print(modelo)
    print(grid.best_score_)
    print(grid.best_estimator_)
    
vetor_modelo = []
vetor_parametros = []

vetor_modelo.append(KNeighborsClassifier())
vetor_parametros.append({'n_neighbors':[75,80,85],
                          'weights':['uniform','distance'],
                          'metric':['manhattan','euclidean']})

vetor_modelo.append(LogisticRegression())
vetor_parametros.append({'C':[0,0.00001,0.0001,0.001,0.01,0.1,1,10],
                          'l1_ratio':[0,0.00001,0.0001,0.001,0.01]})

vetor_modelo.append(DecisionTreeClassifier())
vetor_parametros.append({'criterion':['entropy','gini'],
                          'splitter':['best','random'],
                          'max_depth':[20,30,40],
                          'min_samples_split':[10,20,30],
                          'min_samples_leaf':[30,40,50],
                          'min_weight_fraction_leaf':[0.01,0.1,1]})

vetor_modelo.append(RandomForestClassifier())
vetor_parametros.append({'n_estimators':[1000],
                          'criterion':['entropy'],
                          'max_depth':[None]})

vetor_modelo.append(ExtraTreesClassifier())
vetor_parametros.append({'n_estimators':[400,500,600],
                          'criterion':['gini','entropy'],
                          'max_depth':[1,2,3,4,5,6,None]})

vetor_modelo.append(GradientBoostingClassifier())
vetor_parametros.append({'learning_rate':[1,0,0.1,0.01,0.001],
                          'n_estimators':[100,500,1000],
                          'max_depth':[1,2,3,4,5,6,None]})

vetor_modelo.append(AdaBoostClassifier())
vetor_parametros.append({'learning_rate':[1,0,0.1,0.01,0.001],
                          'n_estimators':[400,500,600]})

vetor_modelo.append(SVC())
vetor_parametros.append({'kernel':['linear','poly','rbf','sigmoid']})


for i in range(len(vetor_modelo)):
    modelo = vetor_modelo[i]
    parametros = vetor_parametros[i]
    treino_grid(modelo,parametros,x,y)
