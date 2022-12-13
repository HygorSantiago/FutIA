import pandas as pd
import numpy as np
from scipy.stats import poisson

import warnings
warnings.filterwarnings("ignore")

times = pd.read_csv('RankingCBF.csv',
                      index_col=0)

def Forca(times):
    a,b = min(times['Pontos']),max(times['Pontos'])
    fa,fb = 0.15,1
    b1 = (fb-fa)/(b-a)
    b0 = fb-b*b1
    times['Força'] = b0+b1*times['Pontos']
    return times

serieA = times.loc[times['Série A'] == 1]
serieA = Forca(serieA)

def Jogo(times,time1,time2):
    
    f1 = 2*times.loc[times.index==time1]['Pontos'].values[0]
    f2 = times.loc[times.index==time2]['Pontos'].values[0]
    
    mgols = 3

    l1 = float(mgols*f1/(f1+f2))
    l2 = float(mgols*f2/(f1+f2))
    
    gols1 = int(np.random.poisson(lam=l1,size=1))
    gols2 = int(np.random.poisson(lam=l2,size=1))
    
    saldo1 = gols1 - gols2
    saldo2 = gols2 - gols1
    
    if gols1 > gols2:
        resultado = 'V'
        pontos1 = 3
        pontos2 = 0
    elif gols1 < gols2:
        resultado = 'D'
        pontos1 = 0
        pontos2 = 3
    elif gols1 == gols2:
        resultado = 'E'
        pontos1 = 1
        pontos2 = 1
        
    return [f'{time1} {gols1} x {gols2} {time2}',time1,time2,pontos1,pontos2,gols1,gols2,saldo1,saldo2,round(l1,2),round(l2,2)]

resultado = Jogo(serieA,'Flamengo - RJ','Atlético - MG')
    
def brasileirao(serieA):
    resultado = []
    for i in serieA.index:
        for j in serieA.index:
            if i != j:
                resultado.append(Jogo(serieA,i,j))
    resultado = pd.DataFrame(resultado,
                             columns=['Placar','Mandante','Visitante','Pontos mandante','Pontos visitante','Gols mandante','Gols visitante','Saldo mandante','Saldo visitante','l1 mandante','l1 visitante'])
    classificacao = pd.DataFrame()
    classificacao.index = serieA.index
    classificacao['Pontos'] = None
    classificacao['Gols feitos'] = None
    classificacao['Gols sofridos'] = None
    
    for i in classificacao.index:
        
        casa = resultado.loc[resultado['Mandante'] == i]
        fora = resultado.loc[resultado['Visitante'] == i]
        
        pontosmandante = casa['Pontos mandante'].sum()
        pontosvisitante = fora['Pontos visitante'].sum()
        pontos = pontosmandante+pontosvisitante
        
        golsfeitosmandante = casa['Gols mandante'].sum()
        golsfeitosvisitante = fora['Gols visitante'].sum()
        golsfeitos = golsfeitosmandante+golsfeitosvisitante
        
        golssofridosmandante = casa['Gols visitante'].sum()
        golssofridosvisitante = fora['Gols mandante'].sum()
        golssofridos = golssofridosmandante+golssofridosvisitante
        
        classificacao['Pontos'][i] = pontos
        classificacao['Gols feitos'][i] = golsfeitos
        classificacao['Gols sofridos'][i] = golssofridos
    
    classificacao['Saldo de gols'] = classificacao['Gols feitos'] - classificacao['Gols sofridos']
    
    return classificacao

c = brasileirao(serieA)
count = 1
for i in range(1000):
    c += brasileirao(serieA)
    count += 1
c = c/count
c = c.sort_values(by='Pontos',
                        ascending=False)

#c.to_excel('media_previsao_brasileirao_2022.xlsx')
