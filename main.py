from rede.camada import Camada
from rede.rede_neural import RedeNeural

from interpretadores import carregar_entrada_txt, carregar_saidas_one_hot


# ======================================
# CRIA REDE
# ======================================
rede = RedeNeural()

tamanho_do_dado=input("Selecione as dimensões do dado \n 1: 12x10 (para identificação de caractéres) \n 2: 2x1 (para algebra booleana)")

match tamanho_do_dado:
    case 1:
        numero_de_camadas=input("Selecione o número de camadas ocultas usados para identificar caractéres")
        #a primeira camada tem que ter exatamente 120 entradas, mas as seguintes podem ter mais ou menos baseado no número de neurônios na camada anterior
    case 2:
        tipo_de_bool=input("Selecione o tipo de problema a resolver \n 1:AND ou OR \n 2:OR \n XOR")
        


match tamanho_do_dado:
    case 1:
       #Camada de entrada deve ter 120 entradas
       rede.adicionar_camada(Camada(
           num_neuronios=1,
           num_entradas=120
       ))
       
        # saída: 26 letras, 26 neurônios para possibilitar one hot
        rede.adicionar_camada(Camada(
            num_neuronios=26,
            num_entradas=60
            )
        )
        
        
    
    
    case 2:
        if(tipo_de_bool==1):
            rede.adicionar_camada(Camada(
                num_neuronios=1,
                num_entradas=2
            ))
        if(tipo_de_bool==2):
            rede.adicionar_camada(Camada(
                num_neuronios=2,
                num_entradas=2
                ))
            rede.adicionar_camada(Camada(
                num_neuronios=1,
                num_entradas=2
            ))
        

numero_de_epocas=input("Selecione o número de épocas ")


rede.adicionar_camada(Camada(num_neuronios=26, num_entradas=60, ativacao=Tanh))

# ======================================
# CARREGAMENTO E DIVISÃO DOS DADOS
# ======================================
dados = carregar_entradas_txt("CARACTERES COMPLETO/X.txt")
saidas = carregar_saidas_one_hot("CARACTERES COMPLETO/Y_letra.txt")

dados = []
saidas = []


entrada_A=carregar_entrada_txt("CARACTERES COMPLETO/X.txt")
dados.append(entrada_A)

saida_A=carregar_saidas_one_hot("CARACTERES COMPLETO/Y_letra.txt")
saidas.append(saida_A)
# exemplo:
# entrada_A = carregar_entrada_txt("CARACTERES COMPLETO/X.txt")
# dados.append(entrada_A)
#saida_A=letra_para_one_hot()
# saidas.append(saida_A)

x_teste = dados[corte:]
y_teste = saidas[corte:]

# ======================================
# TREINO
# ======================================
if len(x_treino) > 0:
    print("Iniciando treinamento...")
    rede.treinar(x_treino, y_treino, epochs=1000, taxa_aprendizado=0.01)
    rede.salvar_pesos("modelos/pesos.json")

# ======================================
# TESTE COMPLETO
# ======================================
print("\nIniciando Teste...")
rede.carregar_pesos("modelos/pesos.json")

    rede.treinar(
        dados,
        saidas,
        epocas=numero_de_epocas,
        taxa_aprendizado=0.1
    )

# loop passando por todas as imagens de teste
for i in range(total_teste):
    entrada = x_teste[i]
    esperado_one_hot = y_teste[i]

    indice_esperado = esperado_one_hot.index(1)

    resultado_previsto = rede.prever(entrada)

    if resultado_previsto == indice_esperado:
        acertos += 1

acuracia = (acertos / total_teste) * 100
print(f"Acurácia no conjunto de teste: {acuracia:.2f}%")
