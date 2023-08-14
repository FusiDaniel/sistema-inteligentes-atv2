import numpy as np
import random
import time
import os

'''
# [ 0] = ⬜"inp_nothing"          **  sente nada (tem nada na casa requisitada)
# [ 1] = 🌀"inp_breeze"           **  sente brisa (uma casa antes de um buraco)
# [ 2] = 😈"inp_danger"           **  sente perigo (casa requisitada/atual tem um Wumpus ou um buraco - morre)
# [ 3] = ✨"inp_flash"            **  sente flash (uma casa antes do ouro ele vê o brilho do ouro)
# [ 4] = 🏆"inp_goal"             **  sente meta (casa requisitada/atual tem ouro - reward, que é a meta)
# [ 5] = 🏁"inp_initial"          **  sente início (casa requisitada/atual é o ponto de partida/saída)
# [ 6] = ⬛"inp_obstruction"      **  sente obstução (mandou request,d e vem obstrução é porque vai colidir em 'd')
# [ 7] = 🦨"inp_stench"           **  sente fedor (uma casa antes de um Wumpus)
# [ 8] = ✨"inp_bf"               **  sente brisa/flash (na casa 'd' tem sinais de brisa e flash)
# [ 9] = ✨"inp_bfs"              **  sente brisa/flash/stench (na casa 'd' tem brisa + flash + fedor)
# [10] = 🌀"inp_bs"               **  sente brisa/stench (na casa 'd' tem brisa + fedor)
# [11] = ✨"inp_fs"               **  sente flash/stench (na casa 'd' tem flash + fedor)
# [12] = ❌"inp_boundary"         **  colidiu com borda (mandou mover forward,d e colidiu com a borda do EnviSim)

# [ 0] = "out_act_grab"         **  ação de pegar/agarrar o ouro (reward)
# [ 1] = "out_act_leave"        **  ação de deixar a caverna (no mesmo local de partida)
# [ 3] = "out_mov_forward"      **  ação de mover adiante
# [11] = "out_rot_left"         **  ação de rotacionar esq.{"rotate":["left",2]}=90°; {"rotate":["left",1]}=45°
# [12] = "out_rot_right"        **  ação de rotacionar esq.{"rotate":["right",2]}=90°; {"rotate":["right",1]}=45°
# [13] = "out_rot_back"         **  ação de rotacionar back.{"rotate":["back",0]}={"rotate":["right",4]}=180°
'''

baseMap = np.array([
    [0, 7, 0, 1, 2, 1, 0, 0, 1, 0],
    [7, 2, 7, 0, 1, 0, 0, 1, 2, 1],
    [0, 7, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 5, 0, 0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 10, 0, 0],
    [1, 2, 1, 0, 0, 0, 7, 2, 7, 0],
    [0, 1, 7, 0, 0, 0, 1, 11, 0, 0],
    [0, 7, 2, 7, 0, 1, 2, 4, 3, 0],
    [0, 0, 7, 0, 0, 0, 1, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
directions = ['u', 'r', 'd', 'l']
dir_dic = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)}
reverse_dir_dic = {(-1, 0): 'u', (1, 0): 'd', (0, -1): 'l', (0, 1): 'r'}
person_dict = {'u': '⬆️', 'd': '⬇️', 'l': '⬅️', 'r': '➡️'}
char_vector = ['⬜', '🌀', '😈', '✨', '🏆', '🏁', '⬛', '🦨', '✨', '✨', '🌀', '✨', '❌']
mapped_movements = [0, 1, 3, 11, 12, 13]


def Criador_de_mapas(dimensao):
    mapa = np.zeros((dimensao, dimensao))
    num_wumpus = dimensao - np.random.randint(0, dimensao-5)
    posi_wumpus = np.zeros((dimensao, 2))
    num_buraco = dimensao - np.random.randint(0, dimensao-4)
    posi_buraco = np.zeros((dimensao, 2))
    posi_ouro = np.zeros((1, 2))
    posi_ouro[0, 0] = np.random.randint(0, dimensao)
    posi_ouro[0, 1] = np.random.randint(0, dimensao)
    inicio = np.zeros((1, 2))
    inicio[0, 0] = np.random.randint(0, dimensao)
    inicio[0, 1] = np.random.randint(0, dimensao)
    # Cria as posicoes aleatorias de buracos e wumpus
    for n in range(dimensao):
        posi_wumpus[n, 0] = np.random.randint(0, dimensao)
        posi_wumpus[n, 1] = np.random.randint(0, dimensao)
        posi_buraco[n, 0] = np.random.randint(0, dimensao)
        posi_buraco[n, 1] = np.random.randint(0, dimensao)
    # confere possiveis casas iguais
    if posi_ouro[0, 0] == inicio[0, 0] and posi_ouro[0, 1] == inicio[0, 1]:
        posi_ouro[0, 1] = posi_ouro[0, 1] + 1
    for i in range(dimensao):
        for j in range(dimensao):
            if posi_wumpus[i, 0] == posi_buraco[j, 0] and posi_wumpus[i, 1] == posi_buraco[j, 1]:
                posi_wumpus[i, 0] = posi_wumpus[i, 0] + 1
    # Percorrendo a matriz para encher com os valores do ouro e obstaculos
    for i in range(dimensao):
        for j in range(dimensao):
            for b in range(num_buraco):
                if i == posi_buraco[b, 0] and j == posi_buraco[b, 1]:
                    mapa[i, j] = 2
            for w in range(num_wumpus):
                if i == posi_wumpus[w, 0] and j == posi_wumpus[w, 1]:
                    mapa[i, j] = 2
            if i == inicio[0, 0] and j == inicio[0, 1]:
                if mapa[i, j] == 2:
                    inicio[0, 0] = inicio[0, 0] + 1
                    inicio[0, 1] = inicio[0, 1] + 1
                else:
                    mapa[i, j] = 5
            if i == posi_ouro[0, 0] and j == posi_ouro[0, 1]:
                if mapa[i, j] == 2:
                    posi_ouro[0, 0] = posi_ouro[0, 0] + 1
                    posi_ouro[0, 1] = posi_ouro[0, 1] + 1
                else:
                    mapa[i, j] = 4
    # Percorrendo a matriz para encher com os valores dos indicadores
    for i in range(dimensao):
        for j in range(dimensao):
            for b in range(num_buraco):
                if i == posi_buraco[b, 0]+1 and j == posi_buraco[b, 1]:
                    mapa[i, j] = 1
                if i == posi_buraco[b, 0]-1 and j == posi_buraco[b, 1]:
                    mapa[i, j] = 1
                if i == posi_buraco[b, 0] and j == posi_buraco[b, 1]+1:
                    mapa[i, j] = 1
                if i == posi_buraco[b, 0] and j == posi_buraco[b, 1]-1:
                    mapa[i, j] = 1
            if i == posi_ouro[0, 0]-1 and j == posi_ouro[0, 1]:
                if mapa[i, j] == 1:
                    mapa[i, j] = 8
                else:
                    mapa[i, j] = 3
            if i == posi_ouro[0, 0]+1 and j == posi_ouro[0, 1]:
                if mapa[i, j] == 1:
                    mapa[i, j] = 8
                else:
                    mapa[i, j] = 3
            if i == posi_ouro[0, 0] and j == posi_ouro[0, 1]+1:
                if mapa[i, j] == 1:
                    mapa[i, j] = 8
                else:
                    mapa[i, j] = 3
            if i == posi_ouro[0, 0] and j == posi_ouro[0, 1]-1:
                if mapa[i, j] == 1:
                    mapa[i, j] = 8
                else:
                    mapa[i, j] = 3
            for w in range(num_wumpus):
                if i == posi_wumpus[w, 0]+1 and j == posi_wumpus[w, 1]:
                    if mapa[i, j] == 1:
                        mapa[i, j] = 10
                    elif mapa[i, j] == 3:
                        mapa[i, j] = 11
                    elif mapa[i, j] == 8:
                        mapa[i, j] = 9
                    else:
                        mapa[i, j] = 7
                if i == posi_wumpus[w, 0]-1 and j == posi_wumpus[w, 1]:
                    if mapa[i, j] == 1:
                        mapa[i, j] = 10
                    elif mapa[i, j] == 3:
                        mapa[i, j] = 11
                    elif mapa[i, j] == 8:
                        mapa[i, j] = 9
                    else:
                        mapa[i, j] = 7
                if i == posi_wumpus[w, 0] and j == posi_wumpus[w, 1]+1:
                    if mapa[i, j] == 1:
                        mapa[i, j] = 10
                    elif mapa[i, j] == 3:
                        mapa[i, j] = 11
                    elif mapa[i, j] == 8:
                        mapa[i, j] = 9
                    else:
                        mapa[i, j] = 7
                if i == posi_wumpus[w, 0] and j == posi_wumpus[w, 1]-1:
                    if mapa[i, j] == 1:
                        mapa[i, j] = 10
                    elif mapa[i, j] == 3:
                        mapa[i, j] = 11
                    elif mapa[i, j] == 8:
                        mapa[i, j] = 9
                    else:
                        mapa[i, j] = 7
           # Percorrendo a matriz para encher com os valores do ouro e obstaculos
    for i in range(dimensao):
        for j in range(dimensao):
            for b in range(num_buraco):
                if i == posi_buraco[b, 0] and j == posi_buraco[b, 1]:
                    mapa[i, j] = 2
            for w in range(num_wumpus):
                if i == posi_wumpus[w, 0] and j == posi_wumpus[w, 1]:
                    mapa[i, j] = 2
            if i == inicio[0, 0] and j == inicio[0, 1]:
                if mapa[i, j] == 2:
                    if i < dimensao:
                        inicio[0, 0] = inicio[0, 0] + 1
                    if j < dimensao:
                        inicio[0, 1] = inicio[0, 1] + 1
                else:
                    mapa[i, j] = 5
            if i == posi_ouro[0, 0] and j == posi_ouro[0, 1]:
                if mapa[i, j] == 2:
                    if i < dimensao:
                        posi_ouro[0, 0] = posi_ouro[0, 0] + 1
                    if j < dimensao:
                        posi_ouro[0, 1] = posi_ouro[0, 1] + 1
                else:
                    mapa[i, j] = 4
    return mapa.astype(int), tuple(inicio[0].astype(int))


def print_state(map, pos, direction):
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if (i, j) == pos:
                print(person_dict[direction], end='  ')
            else:
                print(char_vector[map[i][j]], end=' ')
        print()


def sense(map, pos, dir=-1):
    if dir == -1:
        return map[pos]

    shape = map.shape
    dir = dir_dic[dir]
    dest = (pos[0] + dir[0], pos[1] + dir[1])

    if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
        return 6

    return map[dest]


def senseVector(map, pos, dir, orientation):
    vector = np.zeros((len(orientation), 13), dtype=np.int32)
    for i in range(len(orientation)):
        if orientation[i] == 'f':
            vector[i][sense(map, pos, dir)] = 1
        elif orientation[i] == 'l':
            vector[i][sense(map, pos, directions[(
                directions.index(dir) - 1) % 4])] = 1
        elif orientation[i] == 'r':
            vector[i][sense(map, pos, directions[(
                directions.index(dir) + 1) % 4])] = 1
        elif orientation[i] == 'b':
            vector[i][sense(map, pos, directions[(
                directions.index(dir) + 2) % 4])] = 1
    return vector


def move(map, pos, dir, command, grabbed, win):
    if command == 11:
        dir = directions[directions.index(dir) - 1]
    elif command == 12:
        dir = directions[(directions.index(dir) + 1) % 4]
    elif command == 13:
        dir = directions[(directions.index(dir) + 2) % 4]
    elif command == 3:
        shape = map.shape
        calc_dir = dir_dic[dir]
        dest = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
        if dest[0] >= 0 and dest[1] >= 0 and shape[0] > dest[0] and shape[1] > dest[1]:
            pos = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
            dir = reverse_dir_dic[calc_dir]
    elif command == 0:
        if map[pos] == 4:
            grabbed = True
    elif command == 1:
        if map[pos] == 5 and grabbed:
            win = True
    return pos, dir, grabbed, win, map[pos] == 2

# Talvez essa não seja uma boa ideia porque o algoritmo é recorrente


def has_repeating_items(arr):
    hash_table = set()

    for item in arr:
        if item in hash_table:
            return True
        hash_table.add(item)

    return False

# def has_repeating_sequences(arr, sequence_length):
#     hash_table = set()

#     for i in range(len(arr) - sequence_length + 1):
#         # Get a sequence of specified length
#         sequence = tuple(arr[i:i+sequence_length])

#         if sequence in hash_table:
#             return True
#         hash_table.add(sequence)

#     return False


def infer(vecInpSens: np.int32) -> int:
    return random.choice([0, 1, 3, 11, 12, 13])


def game(infer, movements, enable_print=False, avaliate_game=None):
    map = np.array(baseMap, copy=True)
    pos = (3, 3)
    energy = 200
    dir = random.choice(directions)
    reached, grabbed, win, dead, steppedOnFlash, reachedExit = False, False, False, False, False, False
    dumbness, post_grab_survive = 0, 0
    command_memory = [-1, -1, -1]
    pos_memory = []
    # while energy >= 0 and not has_repeating_sequences(pos_memory[-post_grab_survive:], 5):
    while energy >= 0 and not has_repeating_items(pos_memory):
        # Limpa a tela
        if enable_print:
            os.system('cls' if os.name == 'nt' else 'clear')

        # Recebe os vetores de entradas e retorna saida
        vector = senseVector(map, pos, dir, movements)
        output = infer(vector, grabbed)
        command = mapped_movements[output.index(max(output))]
        command_memory.append(command)
        pos_memory.append((pos, dir))

        # Executa a saida e atualiza estado
        pos, dir, grabbed, win, dead = move(
            map, pos, dir, command, grabbed, win)
        
        # Printa o estado
        if enable_print:
            print_state(map, pos, dir)
            print('Energia: ', energy)
            print('Comando: ', command)
            print('Fitness:', avaliate_game(reached, grabbed, win, dead,
                  steppedOnFlash, reachedExit, dumbness, post_grab_survive))
            if enable_print and grabbed:
                print('Pegou o ouro')
            time.sleep(0.5)
            

        if reachedExit and not win:
            break
        if map[pos] == 4:
            reached = True
        if map[pos] == 3 or map[pos] == 8 or map[pos] == 9 or map[pos] == 11:
            steppedOnFlash = True
        if (vector[0][6] and command == 3) or command_memory[-3:] == [11, 11, 11] or command_memory[-3:] == [12, 12, 12] or command_memory[-2:] == [13, 13] or command_memory[-2:] == [11, 12] or command_memory[-2:] == [12, 11]:
            dumbness += 1
        if (command == 0 and map[pos] != 4) or (command == 0 and map[pos] == 4 and grabbed):
            dumbness += 1
        if (command == 1 and map[pos] != 5) or (command == 1 and not grabbed):
            dumbness += 1
        if len(pos_memory) >= 3 and pos_memory[-3][0] == pos_memory[-2][0] == pos_memory[-1][0]:
            dumbness += 1
        if grabbed:
            if len(pos_memory) > 0 and map[pos_memory[0][0]] != 4:
                pos_memory = []
            if len(pos_memory) >= 3 and pos_memory[-3][0] == pos_memory[-2][0] == pos_memory[-1][0] and map[pos] == 4:
                dumbness += 1
            if map[pos] == 5:
                reachedExit = True

        # Se morreu ou ganhou, para o jogo
        if dead:
            if enable_print:
                print('Morreu')
            break
        if win:
            if enable_print:
                print('Venceu')
            break
        
        energy -= 1
    if enable_print:
        print('Posições repetidas:', has_repeating_items(pos_memory))
        print('win' if win else 'lose')
    # Avalia o jogo
    if avaliate_game:
        return avaliate_game(reached, grabbed, win, dead, steppedOnFlash, reachedExit, dumbness, post_grab_survive)


if __name__ == '__main__':
    # game(infer, ['f', 'l', 'r'], enable_print=True)
    map, pos = Criador_de_mapas(10)
    print_state(map, pos, random.choice(directions))
