#!/usr/bin/python3
import codecs, sys, random, math, time

def sigmoide(valor):
    return 1 / (1 + math.e ** (-valor))

def derivada(valor):
    return sigmoide(valor) * (1 - sigmoide(valor))

def main(filename, arqPesosV="pesosV.txt", arqPesosW="pesosW.txt"):

    linhas = codecs.open(filename).readlines()
    entrada = [] #matriz de entrada, 784 colunas
    saida = [] #matriz de saida, 10 colunas
    qtSaidas = 10 #de 0 a 9
    limiar = 0.5

    #valores serão recebidos a partir dos arquivos de pesos salvos
    qtNeuronios = 0
    qtNeuroniosIntermediarios = 0

    pesosV = [] #matriz de pesos da primeira camada
    pesosW = [] #matriz de pesos da segunda camada

    indice = 0
    for linha in linhas[1:]:
        valores = linha.split(",")
        entrada.append([float(x)/256.0 for x in valores[1:]])
        estaSaida = [0] * 10
        estaSaida[int(valores[0])] = 1
        saida.append(estaSaida)
        indice += 1
        if indice % 1000 == 0:
            print("Importando linha", indice, "de", filename)

    indice = 1
    for linha in codecs.open(arqPesosW).readlines():
        valores = linha.split(",")
        if len(valores) != qtSaidas:
            mensagem = "Erro, linha " + str(indice) + " do arquivo " + str(arqPesosW)
            mensagem += " não tem " + str(qtSaidas) + " elementos"
            raise Exception(mensagem)
        pesosW.append([float(x) for x in valores])
        indice += 1

    indice = 1
    for linha in codecs.open(arqPesosV).readlines():
        valores = linha.split(",")
        if len(valores) != len(pesosW):
            mensagem = "Erro, linha " + str(indice) + " do arquivo " + str(arqPesosV)
            mensagem += " não tem " + str(len(pesosW)) + " elementos"
            raise Exception(mensagem)
        pesosV.append([float(x) for x in valores])
        indice += 1

    qtNeuronios = len(pesosV)
    qtNeuroniosIntermediarios = len(pesosW)

    start = time.time()

    tempo = time.time()
    acertos = 0
    acertosPorNumero = [0] * 10
    tentativasPorNumero = [0] * 10
    errosDeZeroPalpites = 0
    errosDePalpiteErrado = 0
    errosDeMuitosPalpites = 0
    for linha in range(len(entrada)):
        start = time.time()
        z_in = []
        for j in range(qtNeuroniosIntermediarios):
            estaEntrada = entrada[linha]
            z_in.append(sum([estaEntrada[i] * pesosV[i][j] for i in range(qtNeuronios)]))
        z_out = [sigmoide(j) for j in z_in]

        y_in = []
        for k in range(qtSaidas):
            estaEntrada = z_out
            y_in.append(sum([estaEntrada[j] * pesosW[j][k] for j in range(qtNeuroniosIntermediarios)]))
        y_out = [sigmoide(k) for k in y_in]

        estaSaidaEsperada = saida[linha]
        tentativasPorNumero[estaSaidaEsperada.index(1)] += 1

        #retorna um vetor onde os valores de entrada maiores que
        #o limiar são alterados para 1, e os outros 0
        estaResposta = [int(y > limiar) for y in y_out]
        if estaResposta == estaSaidaEsperada:
            acertos += 1
            acertosPorNumero[estaSaidaEsperada.index(1)] += 1
        else:
            palpites = sum(estaResposta)
            if palpites == 0:
                errosDeZeroPalpites += 1
            if palpites == 1:
                errosDePalpiteErrado += 1
            if palpites > 1:
                errosDeMuitosPalpites += 1

        if(linha % 100 == 0 and linha > 0):
            print("Tempo de processamento: ", round(time.time() - tempo, 2) , "s")
            print("Acertos: ", acertos, "/", linha)
            print("Taxa de acerto: ", round(float(acertos*100)/linha, 2), "%")
            for x in range(10):
                print("Acertos onde o número", x, "era esperado:", acertosPorNumero[x], "/", tentativasPorNumero[x], "ou", round(100*acertosPorNumero[x]/tentativasPorNumero[x], 2), "%")
            print("Erros onde o programa não classificou como nenhum número:", errosDeZeroPalpites)
            print("Erros onde o programa classificou como o número errado:", errosDePalpiteErrado)
            print("Erros onde o programa classificou como mais de um número ao mesmo tempo:", errosDeMuitosPalpites)
            print()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Uso: python3 main.py <arquivo de treinamento> <arquivo com pesos 1a camada> <arquivo com pesos 2a camada>")
        print("Se não informados, os pesos serão carregados pelos arquivos padrão pesosV.txt e pesosW.txt")
