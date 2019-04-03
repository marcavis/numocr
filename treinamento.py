#!/usr/bin/python3
import codecs, sys, random, math, time

def sigmoide(valor):
    return 1 / (1 + math.e ** (-valor))

def derivada(valor):
    return sigmoide(valor) * (1 - sigmoide(valor))

def main(filename):

    linhas = codecs.open(filename).readlines()
    entrada = [] #matriz de entrada, 784 colunas
    saida = [] #matriz de saida, 10 colunas
    qtSaidas = 10 #de 0 a 9
    qtNeuronios = len(linhas[0].split(",")) - 1 #784
    #qtNeuroniosIntermediarios = int((10 * 784) ** (0.5)) #media geometrica
    qtNeuroniosIntermediarios = 40 #mas um número menor já basta
    pesosV = [] #matriz de pesos da primeira camada
    pesosW = [] #matriz de pesos da segunda camada
    taxaAprendizado = 0.20


    for linha in linhas[1:]:
        valores = linha.split(",")
        entrada.append([float(x)/256.0 for x in valores[1:]])
        estaSaida = [0] * 10
        estaSaida[int(valores[0])] = 1
        saida.append(estaSaida)

    #criar matriz de pesos da primeira camada com valores aleatórios
    for i in range(qtNeuronios):
        pesosV.append([round(random.random() * 2 - 1, 5) for x in range(qtNeuroniosIntermediarios)])

    for i in range(qtNeuroniosIntermediarios):
        pesosW.append([round(random.random() * 2 - 1, 5) for x in range(qtSaidas)])


    #print(pesosV[0])

    start = time.time()
    erroTotal = 0
    ultimoErroTotal = 0
    #feedforward

    tempo = time.time()
    for iteracao in range(10):
        acertos = 0
        for linha in range(len(entrada)):
        #for linha in range(201):
            #print((time.time() - start) * 1000, "ms")
            start = time.time()
            z_in = []
            for j in range(qtNeuroniosIntermediarios):
                estaEntrada = entrada[linha]
                z_in.append(sum([estaEntrada[i] * pesosV[i][j] for i in range(qtNeuronios)]))
            #z_in = [x for x in z_in] #normalizar para aplicação de sigmoide
            #print (z_in)
            z_out = [sigmoide(j) for j in z_in]
            #print (z_out)



            y_in = []
            for k in range(qtSaidas):
                estaEntrada = z_out
                y_in.append(sum([estaEntrada[j] * pesosW[j][k] for j in range(qtNeuroniosIntermediarios)]))
            #y_in = [round(x, 5) for x in y_in]
            #print()
            #print(y_in)
            y_out = [sigmoide(k) for k in y_in]
            # if random.random() > 0.015:
            #     print (y_in)
            #     q

            estaSaidaEsperada = saida[linha]
            esteErro = sum([abs(estaSaidaEsperada[x] - y_out[x]) for x in range(10)])
            erroTotal += esteErro

            #retorna um vetor onde os valores de entrada maiores que
            #o limiar são alterados para 1, e os outros 0
            estaResposta = [int(y > 0.7) for y in y_out]
            if estaResposta == estaSaidaEsperada:
                acertos += 1

            if(linha % 100 == 0 and linha > 0):
                #print(estaSaidaEsperada)
                #print([round(y,4) for y in y_out])
                #print(estaResposta)

                #print("Este erro: ", esteErro)
                print("Tempo de processamento: ", round(time.time() - tempo, 2) , "s")
                print("Erro fracionado das últimos 100 linhas: ", linha, ": ", erroTotal - ultimoErroTotal)
                print("Acertos nessa epoca: ", acertos, "/", linha)
                print("Taxa de acerto: ", round(float(acertos*100)/linha, 2), "%")
                print()
                ultimoErroTotal = erroTotal
            termoInfErroFinal = [(estaSaidaEsperada[k] - y_out[k]) * derivada(y_in[k]) for k in range(qtSaidas)]

            #print(estaSaidaEsperada[0] - y_out[0])
            delta_w = []
            for j in range(qtNeuroniosIntermediarios):
                delta_w.append([taxaAprendizado * termoInfErroFinal[k] * z_out[j] for k in range(qtSaidas) ])

            delta_in = []
            for j in range(qtNeuroniosIntermediarios):
                delta_in.append(sum([termoInfErroFinal[k] * pesosW[j][k] for k in range(qtSaidas)]))

            #print(delta_in)

            termoInfErroInicial = [delta_in[j] * derivada(z_in[j]) for j in range(qtNeuroniosIntermediarios)]


            delta_v = []
            for i in range(qtNeuronios):
                delta_v.append([taxaAprendizado * termoInfErroInicial[j] * entrada[linha][i] for j in range(qtNeuroniosIntermediarios) ])

            #print(delta_v)
            #print(linha, sum([sum(x) for x in delta_v]))


            #atualização de pesos

            for i in range(len(pesosV)):
                pesosV[i] = [x + y for (x,y) in zip(pesosV[i], delta_v[i])]

            for j in range(len(pesosW)):
                pesosW[j] = [x + y for (x,y) in zip(pesosW[j], delta_w[j])]


            # total = 0
            # for x in pesosV: total += sum(x)
            # print(total)
            #print(pesosW[10][2])
    arquivoPesosV = codecs.open("pesosV.txt", 'w')
    for i in range(len(pesosV)):
        inserir = pesosV[i] = [round(v, 5) for v in pesosV[i]]
        arquivoPesosV.write(str(inserir)[1:-1] + "\n")

    arquivoPesosW = codecs.open("pesosW.txt", 'w')
    for j in range(len(pesosW)):
        inserir = pesosW[j] = [round(w, 5) for w in pesosW[j]]
        arquivoPesosW.write(str(inserir)[1:-1] + "\n")

    arquivoPesosV.close()
    arquivoPesosW.close()
    print(pesosV)
    print()
    print()
    print()
    print()
    print()
    print(pesosW)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Uso: python3 main.py <arquivo de treinamento>")
