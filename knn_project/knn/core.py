import pandas

def carregar_csv(filepath):
    df = pandas.read_csv(filepath) #cria um dataframe
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    X = df.drop(columns=["Species"]) #atributos
    y = df["Species"]  #espécies
    #print(df.head())
    return X,y

X, y = carregar_csv("data/Iris.csv")
df_completo = pandas.concat([X,y], axis=1)

#print(X.head())
#print(y.head())

#print(X[:])
#print(y[:])


random_state = 30 #definindo uma seed para replicabilidade
train_frac = 0.7 #porcentagem usada para treino

df_embaralhado = df_completo.sample(frac=1, random_state=random_state).reset_index(drop=True) 
#retorna todas as linhas embaralhadas
#usando o random_state para replicar se necessário
#também reinicia os indices

train_size = int(train_frac*len(df_embaralhado))



