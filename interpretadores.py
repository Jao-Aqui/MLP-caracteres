def carregar_entradas_txt(arquivo):
    dados_totais = []

    with open(arquivo, "r") as f:
        conteudo = f.read()

    valores = conteudo.split(",")

    entrada = []

    for v in valores:

        valor = int(v.strip())

        # validação
        if valor not in (-1, 1):

            raise ValueError(
                f"Valor inválido na entrada: {valor}"
            )

        # converte:
        # -1 -> 0
        #  1 -> 1
        entrada.append(
            0 if valor == -1 else 1
        )

    if len(entrada) != 120:

        raise ValueError(
            f"Esperado 120 valores, recebido {len(entrada)}"
        )

            if len(entrada) == 120:
                dados_totais.append(entrada)
            elif len(entrada) > 0:
                print(f"Aviso: Linha ignorada. Esperado 120, recebido {len(entrada)}")

    return dados_totais

def letra_para_one_hot(letra):

    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    letra = letra.upper()

    if letra not in alfabeto:

        raise ValueError(
            f"Letra inválida: {letra}"
        )

    # sigmoid trabalha melhor com 0/1
    vetor = [0] * 26

    indice = alfabeto.index(letra)

    vetor[indice] = 1

    return vetor


def carregar_saidas_one_hot(arquivo):

    saidas = []

    with open(arquivo, "r") as f:

        for linha in f:

            letra = linha.strip()

            # ignora linhas vazias
            if letra != "":

                saidas.append(letra_para_one_hot(letra))

    return saidas
