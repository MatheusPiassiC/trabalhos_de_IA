from collections import deque
import heapq
import time
import random

OBJETIVO = (1,2,3,8,0,4,7,6,5) #configuração desejada do tabuleiro
MOVIMENTOS = {'cima': -3, 'baixo': 3, 'esquerda': -1, 'direita': 1}
ORDEM_DOS_MOVIMENTOS = ('cima', 'esquerda', 'direita', 'baixo')

def movimento_valido(posicao, movimento): #verifica se um movimento é válido
    if movimento == 'esquerda' and posicao % 3 == 0: return False  #impossível ir para a esquerda
    if movimento == 'direita' and posicao % 3 == 2: return False   #impossível ir para a direita
    if movimento == 'cima' and posicao < 3: return False           #impossível ir para cima
    if movimento == 'baixo' and posicao > 5: return False          #impossivel ir para baixo
    return True #o movimento é válido

def troca(estado, i, j): #troca um numero de posição com outro (no caso, um espaço em branco)
    lista = list(estado)
    lista[i], lista[j] = lista[j], lista[i]
    return tuple(lista)

def sucessores(estado):
    posicao = estado.index(0) #retorna o indice do espaço em branco
    for movimento in ORDEM_DOS_MOVIMENTOS: #verifica quais movimentos são válidos
        if movimento_valido(posicao, movimento): 
            nova_pos = posicao + MOVIMENTOS[movimento]
            novo_estado = troca(estado, posicao, nova_pos)
            yield novo_estado, movimento #retorna um gerador, um objeto iteravel que produz valores sobre demanda

def inversoes(seq):
    return sum(1 for i in range(len(seq)) for j in range(i+1, len(seq)) if seq[i] > seq[j])

def resolvivel(estado): #verifica se há uma solução para um determinado estado incial
    seq_estado = [x for x in estado if x != 0] #cria sequência do estado inicial ignorando o 0 (casa vazia)
    seq_objetivo = [x for x in OBJETIVO if x != 0] #cria um sequencia do objetivo, ignorando o 0
    return (inversoes(seq_estado) % 2) == (inversoes(seq_objetivo) % 2) #verifica a paridade de inversões 

def reconstruir(origens, fim): #reconstroi o caminho feito para chegar a um estado, segundo as origens
    caminho = []
    atual = fim
    while atual in origens:
        atual, movimento = origens[atual]
        caminho.append(movimento)
    caminho.reverse()
    return caminho

def manhattam(s): #heurística manhatan, calcula a distância de cada número em relação ao objetivo
    total = 0
    for i, v in enumerate(s):
        if v == 0:
            continue
        # encontra a posição alvo da peça v na tupla OBJETIVO
        target_idx = OBJETIVO.index(v)
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(target_idx, 3)
        total += abs(r1 - r2) + abs(c1 - c2)
    return total #soma das distâncias manhattan

def bfs(estado_inicial): #algoritmo de resolução usando bfs
    if estado_inicial == OBJETIVO: #verifica se o estado inicial já é o objetivo
        return [], 0, 0.0
    if not resolvivel(estado_inicial): #verifica se há uma solução a partir do estado inicial
        return None, 0, 0.0
    
    t0 = time.time()

    fronteira = deque([estado_inicial]) #estrutura de dados padrão da bfs, guarda os estados descobertos
    fronteira_set = {estado_inicial}    #estrutura auxiliar

    origens = {}
    fechado = set() #estados explorados
    nos = 0         #nós explorados

    while fronteira:
        atual = fronteira.popleft() #retira o estado ataul da pilha
        fronteira_set.remove(atual) #e também da estrutura auxiliar
        fechado.add(atual) #adiciona o estado atual ao conjunto dos estados explorados
        nos += 1

        for (estado,movimento) in sucessores(atual): #itera sobre todos as possibilidades a partir de um estado
            if estado in fechado or estado in fronteira_set: #verifica se o estado possível está no conjunto fechado ou na fronteira
                continue                      #se sim, pula para a próxima iteração, pois estes estados não precisam de ser explorados
            origens[estado] = (atual, movimento) 
            if estado == OBJETIVO: #se o estado for o objetivo, reconstroi o caminho e retorna
                return reconstruir(origens, estado), nos, time.time() - t0 
            fronteira.append(estado) #se não for o objetivo, adiciona à fronteira
            fronteira_set.add(estado)

    return None, nos, time.time() - t0

def astar(estado_inicial): #algoritmo de resolução usando A*
    if estado_inicial == OBJETIVO: #verifica se o estado inicial já é o objetivo
        return [], 0, 0.0
    if not resolvivel(estado_inicial): #verifica se há uma solução a partir do estado inicial
        return None, 0, 0.0
    
    t0 = time.time()
    heap_abertos = [] #fila de prioridade, implementado com o heapq
    contador = 0
    f_inicial = manhattam(estado_inicial) #distância do estado inical do objetivo
    heapq.heappush(heap_abertos, (f_inicial, contador, estado_inicial)) #heap, menor item (considerando o atributo f_inicial) na posição 0
                                                                        #cada entrada é uma tupla (f, contador, estado)     
    g_score = {estado_inicial: 0} #guarda o custo real mínimo do inicio até cada estado
    origens = {}    #guarda a origem de cada estado para ajudar a recriar o caminho
    fechado = set() #estados já expandidos
    nos = 0         #contador de nos/estados expandidos

    while heap_abertos:
        _, _, atual = heapq.heappop(heap_abertos) #pega o item de menor

        if atual in fechado: #se o estado atual está fechado, pula para a próxima iteração
            continue

        if atual == OBJETIVO:
            return reconstruir(origens, atual), nos, time.time() - t0
        
        fechado.add(atual)
        nos += 1

        for estado, movimento in sucessores(atual): #itera sobre os movimentos possíveis a partir do estado atual
            tentativa_g = g_score[atual] + 1 #custo do caminho atual, gscore do no atual + 1 movimento

            if estado in fechado and tentativa_g >= g_score.get(estado, float('inf')):
                continue #se o vizinho já foi fechado com custo melhor ou igual, pulamos a iteração

            #se encontramos um caminho melhor atualizamos as estruturas
            if tentativa_g < g_score.get(estado, float('inf')):
                origens[estado] = (atual, movimento) #guarda a origem e qual movimento levou até aqui
                g_score[estado] = tentativa_g        #atualiza o custo real do estado
                f = tentativa_g + manhattam(estado)  #estimativa do custo total passando pelo estado atual
                contador += 1
                heapq.heappush(heap_abertos, (f, contador, estado)) #adiciona o estado ao heap

    return None, nos, time.time() - t0


def gerar_estado_aleatorio(passos=10, seed=None):
    """Gera um estado inicial aplicando `passos` movimentos aleatórios a partir de `OBJETIVO`.
    Assim garantimos que o estado é solucionável em relação a `OBJETIVO`.
    """
    if seed is not None:
        random.seed(seed)
    estado = OBJETIVO
    for _ in range(passos):
        succ = list(sucessores(estado))
        # succ é lista de (novo_estado, movimento)
        if not succ:
            break
        estado, _ = random.choice(succ)
    return estado


if __name__ == '__main__':
    random.seed()
    TESTS = 3
    PASSOS = 8
    for i in range(TESTS):
        print(f"\n=== Teste {i+1}/{TESTS} ===")
        estado_inicial = gerar_estado_aleatorio(passos=PASSOS)
        print('Estado inicial:', estado_inicial)
        print('Objetivo     :', OBJETIVO)
        print('Solvável?    :', resolvivel(estado_inicial))

        print('\nExecutando BFS...')
        caminho_bfs, nos_bfs, tempo_bfs = bfs(estado_inicial)
        if caminho_bfs is None:
            print('BFS: sem solução')
        else:
            print(f'BFS: passos={len(caminho_bfs)}, nós_expandidos={nos_bfs}, tempo={tempo_bfs:.4f}s')

        print('\nExecutando A*...')
        caminho_astar, nos_astar, tempo_astar = astar(estado_inicial)
        if caminho_astar is None:
            print('A*: sem solução')
        else:
            print(f'A*: passos={len(caminho_astar)}, nós_expandidos={nos_astar}, tempo={tempo_astar:.4f}s')

    print('\nTodos os testes concluídos.')





    
