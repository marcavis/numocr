#!/usr/bin/python3
import codecs, sys, random, math

def sigmoide(valor):
    return 1 / (1 + math.e ** (-valor))

def derivada(valor):
    return sigmoide(valor) * (1 - sigmoide(valor))

def main(filename):
    linhas = codecs.open(filename).readlines()
    entrada = [] #matriz de entrada, 784 colunas
    saida = [] #matriz de saida, 10 colunas
    qtSaidas = 10 #de 0 a 9
    qtNeuronios = len(linhas[0].split(",")) - 1
    qtNeuroniosIntermediarios = int((10 * 784) ** (1/2)) #media geometrica
    pesosV = [] #matriz de pesos da primeira camada
    pesosW = [] #matriz de pesos da segunda camada
    taxaAprendizado = 0.08


    for linha in linhas[1:]:
        valores = linha.split(",")
        entrada.append([int(x) for x in valores[1:]])
        estaSaida = [0] * 10
        estaSaida[int(valores[0])] = 1
        saida.append(estaSaida)

    #criar matriz de pesos da primeira camada com valores aleatórios
    for i in range(qtNeuronios):
        pesosV.append([round(random.random(),5) for x in range(qtNeuroniosIntermediarios)])

    for i in range(qtNeuroniosIntermediarios):
        pesosW.append([round(random.random(),5) for x in range(qtSaidas)])

    #feedforward
    for linha in range(len(entrada)):
        z_in = []
        for j in range(qtNeuroniosIntermediarios):
            estaEntrada = entrada[linha]
            z_in.append(sum([estaEntrada[i] * pesosV[i][j] for i in range(qtNeuronios)]))
        z_in = [round(x/16384.0, 5) for x in z_in] #normalizar para aplicação de sigmoide
        print (z_in)
        z_out = [sigmoide(j) for j in z_in]
        print (z_out)

        y_in = []
        for k in range(qtSaidas):
            estaEntrada = z_out
            y_in.append(sum([estaEntrada[j] * pesosW[j][k] for j in range(qtNeuroniosIntermediarios)]))
        y_in = [round(x/32.0, 5) for x in y_in]
        print()
        print(y_in)
        y_out = [sigmoide(k) for k in y_in]
        print (y_out)

        estaSaida = saida[linha]
        delta = [(estaSaida[k] - y_out[k]) * derivada(y_in[k]) for k in range(qtSaidas)]
        #print(delta)
        delta_w = []
        for j in range(qtNeuroniosIntermediarios):
            delta_w.append([taxaAprendizado * delta[k] * z_out[j] for k in range(qtSaidas) ])

        delta_in = []
        for j in range(qtNeuroniosIntermediarios):
            delta_in.append(sum([delta[k] * pesosW[j][k] for k in range(qtSaidas)]))

        print(delta_in)

        delta_j = [delta_in[j] * derivada(z_in[j]) for j in range(qtNeuroniosIntermediarios)]

        delta_v = []
        for i in range(qtNeuronios):
            delta_v.append([taxaAprendizado * 10* delta_j[j] * entrada[linha][i] for j in range(qtNeuroniosIntermediarios) ])

        print(delta_v)
        ssa



if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Uso: python3 main.py <arquivo de treinamento>")
