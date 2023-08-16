import numpy as np
import random
import time
import os
from math import *

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

Simpificação dos inputs:
# [ 0][ 1][ 7][10] = ⬜"inp_nothing"          **  sente nada (tem nada na casa requisitada)
# [ 2][ 6][12] = 😈"inp_danger"           **  sente perigo (casa requisitada/atual tem um Wumpus ou um buraco - morre)
# [ 3][ 8][ 9][11] = ✨"inp_flash"            **  sente flash (uma casa antes do ouro ele vê o brilho do ouro)
# [ 4] = 🏆"inp_goal"             **  sente meta (casa requisitada/atual tem ouro - reward, que é a meta)
# [ 5] = 🏁"inp_initial"          **  sente início (casa requisitada/atual é o ponto de partida/saída)

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
basePos = (3, 3)

directions = ['u', 'r', 'd', 'l']
dir_dic = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)}
reverse_dir_dic = {(-1, 0): 'u', (1, 0): 'd', (0, -1): 'l', (0, 1): 'r'}
person_dict = {'u': '⬆️', 'd': '⬇️', 'l': '⬅️', 'r': '➡️'}
char_vector = ['⬜', '🌀', '😈', '✨', '🏆', '🏁', '⬛', '🦨', '✨', '✨', '🌀', '✨', '❌']
mapped_movements = [0, 1, 3, 11, 12, 13]

# Aprender a não morrer
danger_map = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 1, 2, 1],
    [0, 1, 2, 1, 2],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
])
# Aprender a chegar no ouro, pegar e sair
goal_map = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 3, 4, 3, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0],
])
# Aprender a sair da caverna quando tem ouro
start_map = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])


def random_map(dimensao):
    map = np.zeros((dimensao, dimensao), dtype=int)
    num_danger = dimensao
    num_danger_per_quadrant = int(num_danger / 4)
    danger_positions = []

    # Distribui perigos aleatoriamente nos quadrantes do mapa
    for i in range(2):
        for j in range(2):
            start_row = i * (dimensao // 2)
            end_row = (i + 1) * (dimensao // 2)
            start_col = j * (dimensao // 2)
            end_col = (j + 1) * (dimensao // 2)
            quadrant_indices = np.random.choice(
                range(start_row, end_row), num_danger_per_quadrant, replace=False)
            for idx in quadrant_indices:
                col_idx = np.random.choice(range(start_col, end_col))
                danger_positions.append((idx, col_idx))

    wumpus_positions = danger_positions[:num_danger//2]
    hole_positions = danger_positions[num_danger//2:]

    # Popula mapa com wumpus e buracos
    for i in range(len(wumpus_positions)):
        map[wumpus_positions[i][0]][wumpus_positions[i][1]] = 2
    for i in range(len(hole_positions)):
        map[hole_positions[i][0]][hole_positions[i][1]] = 2

    # Gera posição do ouro e do início
    gold_position = np.random.randint(0, dimensao, size=(1, 2))
    while len(np.unique(gold_position, axis=0)) != 1 or np.any(np.all(gold_position == wumpus_positions, axis=1)) or np.any(np.all(gold_position == hole_positions, axis=1)):
        gold_position = np.random.randint(0, dimensao, size=(1, 2))
    start_position = np.random.randint(0, dimensao, size=(1, 2))
    while len(np.unique(start_position, axis=0)) != 1 or np.any(np.all(start_position == wumpus_positions, axis=1)) or np.any(np.all(start_position == hole_positions, axis=1)) or np.all(start_position == gold_position):
        start_position = np.random.randint(0, dimensao, size=(1, 2))

    # Posiciona ouro e início no mapa
    map[gold_position[0][0]][gold_position[0][1]] = 4
    map[start_position[0][0]][start_position[0][1]] = 5

    wumpus_indicators_positions = []
    holes_indicators_positions = []
    gold_indicators_positions = []

    # Gera indicadores de perigo e ouro
    for item in [(wumpus_positions, wumpus_indicators_positions), (hole_positions, holes_indicators_positions), (gold_position, gold_indicators_positions)]:
        for i in range(len(item[0])):
            row = item[0][i][0]
            col = item[0][i][1]
            if row > 0:
                item[1].append((row - 1, col))
            if row < dimensao - 1:
                item[1].append((row + 1, col))
            if col > 0:
                item[1].append((row, col - 1))
            if col < dimensao - 1:
                item[1].append((row, col + 1))

    all_indicators_positions = []
    
    for i in wumpus_indicators_positions:
        if i in holes_indicators_positions and i in gold_indicators_positions:
            all_indicators_positions.append((i, 8))
            holes_indicators_positions.pop(holes_indicators_positions.index(i))
            gold_indicators_positions.pop(gold_indicators_positions.index(i))
        elif i in gold_indicators_positions:
            all_indicators_positions.append((i, 11))
            gold_indicators_positions.pop(gold_indicators_positions.index(i))
        elif i in holes_indicators_positions:
            all_indicators_positions.append((i, 10))
            holes_indicators_positions.pop(holes_indicators_positions.index(i))
        else:
            all_indicators_positions.append((i, 7))
    for i in holes_indicators_positions:
        if i in gold_indicators_positions:
            all_indicators_positions.append((i, 8))
            gold_indicators_positions.pop(gold_indicators_positions.index(i))
        else:
            all_indicators_positions.append((i, 1))
    for i in gold_indicators_positions:
        all_indicators_positions.append((i, 3))

    # Posiciona indicadores no mapa
    for i in all_indicators_positions:
        if map[i[0]] == 0:
            map[i[0]] = i[1]

    return map, (start_position[0][0], start_position[0][1])

# Método que imprime o estado do jogo
def print_state(map, pos, direction):
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if (i, j) == pos:
                print(person_dict[direction], end='  ')
            else:
                print(char_vector[map[i][j]], end=' ')
        print()

# Método que sente o ambiente em uma direção
def sense(map, pos, dir=-1):
    if dir == -1:
        return map[pos]

    shape = map.shape
    dir = dir_dic[dir]
    dest = (pos[0] + dir[0], pos[1] + dir[1])

    if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
        return 6

    return map[dest]

# Método que cria um vetor com os dados sentidos em cada direção fornecida
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

# Método que move o persoangem no mapa a partir da posição, comando e direção
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

# Instancía um jogo e retorna o fitness
def game(infer, movements, enable_print=False, avaliate_game=None, train_step=2, print_delay=0.1):
    
    fitness = 0
    # TREINAMENTO INICIAL
    # Aprender a não morrer 
    loop = [
        ((3,2), 'u', [11, 12, 13]),
        ((3,2), 'l', [3, 11, 13]),
        ((3,2), 'r', [3, 12, 13]),
        ((2,3), 'u', [13]),
        ((2,3), 'l', [11, 13]),
        ((2,3), 'r', [12, 13]),
        ]
    for pos, dir, expected_commands in loop:
        vector = senseVector(danger_map, pos, dir, movements)
        output = infer(vector, danger_map[pos] == 4, random.choice([True, False]), danger_map[pos] == 5)
        command = mapped_movements[output.index(max(output))]
        fitness += 1 if command in expected_commands else 0
    
    # Aprender a chegar no ouro, pegar e sair
    loop = [
            # no ouro
        ((2,2), 'u', [0], False),
        ((2,2), 'u', [11,12,13], True),
        ((2,2), 'l', [3,11,13], True),
        ((2,2), 'r', [3,12,13], True),
        ((2,2), 'd', [3,11,12,13], True),
        # aos arredores dos brilhos
        ((4,2), 'u', [3], False),
        ((4,2), 'l', [12], False),
        ((4,2), 'r', [11], False),
        ((3,3), 'u', [3], False),
        ((3,3), 'l', [3], False),
        ((3,3), 'r', [11], False),
        ((3,3), 'u', [3,11,12,13], True),
        ((3,3), 'l', [3,11,12,13], True),
        ((3,3), 'r', [3,11,12,13], True),
        # aos arredores do ouro
        ((3,2), 'u', [3], False),
        ((3,2), 'l', [12], False),
        ((3,2), 'r', [11], False),
        ((3,2), 'u', [3,11,12,13], True),
        ((3,2), 'l', [3,11,12,13], True),
        ((3,2), 'r', [3,11,12,13], True),
    ]
    for index, (pos, dir, expected_commands, grabbed) in enumerate(loop):
        vector = senseVector(goal_map, pos, dir, movements)
        output = infer(vector, goal_map[pos] == 4, grabbed, goal_map[pos] == 5)
        command = mapped_movements[output.index(max(output))]
        if index == 0:
            fitness += 5 if command in expected_commands else -5
        else:
            fitness += 1 if command in expected_commands else 0
    
    # Aprender a sair da caverna quando tem ouro
    loop = [
        # quando tem ouro
        ((2,2), 'u', [1], True),
        ((3,2), 'u', [3], True),
        ((3,2), 'l', [12], True),
        ((3,2), 'r', [11], True),
        ((2,2), 'r', [1], True),
        ((2,2), 'l', [1], True),
        ((2,2), 'd', [1], True),
        # quando não tem ouro
        ((3,2), 'u', [3,11,12,13], False),
        ((3,2), 'r', [3,11,12,13], False),
        ((3,2), 'l', [3,11,12,13], False),
        ((2,2), 'u', [11,12,13], False),
        ((2,2), 'l', [3,11,13], False),
        ((2,2), 'r', [3,12,13], False),
    ]
    for index, (pos, dir, expected_commands, grabbed) in enumerate(loop):
        vector = senseVector(start_map, pos, dir, movements)
        output = infer(vector, start_map[pos] == 4, grabbed, start_map[pos] == 5)
        command = mapped_movements[output.index(max(output))]
        if index == 0:
            fitness += 5 if command in expected_commands else -5
        else:
            fitness += 1 if command in expected_commands else 0

    # APRENDER A ANDAR PELO MAPA
    energy = 200
    dir = random.choice(directions)
    reachedGoal, grabbed, win, dead, steppedOnFlash, reachedExit = False, False, False, False, False, False
    dumbness = 0
    command_memory = [-1, -1, -1]
    pos_memory = []

    map, pos = np.array(baseMap, copy=True), basePos

    if enable_print:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_state(map, pos, dir)
        time.sleep(1)
    while energy >= 0 and not (win or dead) and train_step == 2:
        if enable_print:
            os.system('cls' if os.name == 'nt' else 'clear')
        # Sente o ambiente
        vector = senseVector(map, pos, dir, movements)
        # Consome a rede neural com o dados sentidos
        output = infer(vector, map[pos] == 4, grabbed, map[pos] == 5)
        command = mapped_movements[output.index(max(output))]

        command_memory.append(command)
        pos_memory.append((pos, dir))

        # Executa a saida e atualiza estado
        pos, dir, grabbed, win, dead = move(
            map, pos, dir, command, grabbed, win)

        # Imprime o estado
        if enable_print:
            print_state(map, pos, dir)
            print('Energia: ', energy)
            if win:
                print('🏆 VENCEU!!!')
            elif dead:
                print('😈 PERDEU')
            elif grabbed:
                print('🏆')
            # if grabbed and command != 0 and command != 3: time.sleep(100)

            time.sleep(print_delay)

        # Penalidades e fim de jogo
        if reachedExit and not win:
            break
        if map[pos] == 4:
            reachedGoal = True
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

        energy -= 1

    # Avalia o jogo
    if avaliate_game and train_step == 2:
        fitness += avaliate_game(reachedGoal, grabbed, win, dead, steppedOnFlash, reachedExit, dumbness)
    return fitness

# Gerador de mapa aleatório
def generateMap():
    size = int(input("Digite o tamanho n do mapa (n x n): "))
    map, pos = random_map(size)
    print_state(map, pos, random.choice(directions))

    print("Para usar o mapa, substitua o baseMap e o basePos no arquivo game.py pelos seguintes")

    print("baseMap = np.array(")
    print(np.array2string(map, separator=', '))
    print(")")

    print(f'basePos = {pos}')

