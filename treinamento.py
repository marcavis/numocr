#!/usr/bin/python3
import codecs, sys, random

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
        pesosV.append([round(random.random(),2) for x in range(qtNeuroniosIntermediarios)])

    for i in range(qtNeuroniosIntermediarios):
        pesosW.append([round(random.random(),2) for x in range(qtSaidas)])

    #feedforward
    estaEntrada = entrada[0]
    z_in = sum([estaEntrada[x] * pesosV[x][0] for x in range(len(estaEntrada))])
    print (estaEntrada)
    print ([x[0] for x in pesosV])
    print ([estaEntrada[x] * pesosV[x][0] for x in range(len(estaEntrada))])
    print (z_in)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Uso: python3 main.py <arquivo de treinamento>")
