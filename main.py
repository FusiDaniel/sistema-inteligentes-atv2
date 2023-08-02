
import numpy as np
import random
import time

'''
0 = vazio
1 = atencao
2 = morte
3 = brilho
4 = ouro
5 = entrada/saida
'''

import os
def clear():
    os.system('cls')

baseMap = np.array([
    [0,1,0,1,2,1,0,0,1,0],
    [1,2,1,0,1,0,0,1,2,1],
    [0,1,0,0,0,0,0,1,1,0],
    [0,0,0,5,0,0,1,2,1,0],
    [0,1,0,0,0,0,0,1,0,0],
    [1,2,1,0,0,0,1,2,1,0],
    [0,1,1,0,0,0,1,3,0,0],
    [0,1,2,1,0,1,2,4,3,0],
    [0,0,1,0,0,0,1,3,0,0],
    [0,0,0,0,0,0,0,0,0,0]
])
directions = ['c','d','b','e']
dir_dic = {'c' : (-1,0), 'd' : (0,1), 'b' : (1,0), 'e' : (0,-1)}
reverse_dir_dic = {(-1, 0): 'c', (0, 1): 'd', (1, 0): 'b', (0, -1): 'e'}
person_dict = {'c' : '⬆️', 'b' : '⬇️', 'd' : '➡️', 'e' : '⬅️'}
char_vector = ['⬜','🌀','😈','✨','🏆','🏁','⬛','❓']
mapped_movements = {11: 'l', 12: 'r', 13: 'b', 3: 'f', 0: 'g', 1: 'v'}

def print_state(map, pos, direction, grabbed):
  for i in range(map.shape[0]):
    for j in range(map.shape[1]):
      if (i, j) == pos:
        print(person_dict[direction], end='  ')
      elif map[i][j] == 4 and grabbed:
        print(char_vector[6], end=' ')
      else:
        print(char_vector[map[i][j]], end=' ')
    print()
  if grabbed:
    print(char_vector[4], end='')

def sense(map, pos, dir):
  shape = map.shape
  dir = dir_dic[dir]
  dest = (pos[0] + dir[0], pos[1] + dir[1])

  if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
    return 6
  return map[dest]

def move(map, pos, dir, command, grabbed, win):
  if command == 'r':
    dir = directions[(directions.index(dir) + 1) % 4]
  elif command == 'l':
    dir = directions[directions.index(dir) - 1]
  elif command == 'b':
    dir = directions[(directions.index(dir) + 2) % 4]
  elif command == 'f':
    shape = map.shape
    calc_dir = dir_dic[dir]
    dest = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
    if dest[0] >= 0 and dest[1] >= 0 and shape[0] > dest[0] and shape[1] > dest[1]:
      pos = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
      dir = reverse_dir_dic[calc_dir];
  elif command == 'g':
    if map[pos] == 4:
      grabbed = True
  elif command == 'v':
    if map[pos] == 5 and grabbed:
      win = True
  return pos, dir, grabbed, win
def check_death(map, pos):
  if map[pos] == 2:
    clear()
    return True
  return False

def make_decision(probabilities):
    items_list = list(probabilities.keys())
    probabilities_list = list(probabilities.values())

    return random.choices(items_list, weights=probabilities_list, k=1)[0]

def hasFlash(vecInpSens: np.int32, i: int) -> bool:
    return vecInpSens[i][3] == 1
def hasGoal(vecInpSens: np.int32, i: int) -> bool:
    return vecInpSens[i][4] == 1
def hasObstacle(vecInpSens: np.int32, i: int) -> bool:
    return vecInpSens[i][2] == 1 or vecInpSens[i][6] == 1

last_choice = -1
reached_goal = False

def infer(vecInpSens: np.int32) -> int:
    global last_choice, memory, reached_goal
    outy = -1  # por default, o índice de saída é um índice de erro
    
    if np.sum(vecInpSens) == 0 :  # se num_input_bits for zero
        return outy  # retorna erro (-1)
    else:
        if (reached_goal):
            outy = 0
        elif (hasFlash(vecInpSens, 0)):
            outy = 3
        elif (hasFlash(vecInpSens, 1) or hasGoal(vecInpSens, 1)):
            outy = 11
        elif (hasFlash(vecInpSens, 2) or hasGoal(vecInpSens, 2)):
            outy = 12
        elif (hasGoal(vecInpSens, 0)):
            outy = 3
            reached_goal = True

        elif (hasObstacle(vecInpSens, 0)):
            if (hasObstacle(vecInpSens, 1)):
                outy = make_decision({12: 6, 13: 1})
            elif (hasObstacle(vecInpSens, 2)):
                outy = make_decision({11: 6, 13: 1})
                    
            else:
                if (last_choice == 11) :
                    outy = make_decision({11: 8, 13: 1})
                if (last_choice == 12) :
                    outy = make_decision({11: 8, 13: 1})
                else:
                    outy = make_decision({11: 4, 12: 4, 13: 1})
        else:
            if (last_choice == 11 or last_choice == 12):
                outy = 3
            elif (hasObstacle(vecInpSens, 1)):
                outy = make_decision({3: 7, 12: 1})
            elif (hasObstacle(vecInpSens, 2)):
                outy = make_decision({3: 7, 11: 1})
            else:
                outy = make_decision({3: 7, 11: 1, 12: 1})
    last_choice = outy
    return outy


def senseVector( map, pos, dir):
  vector = np.zeros((3, 13), dtype=np.int32)
  vector[0][sense(map, pos, dir)] = 1
  vector[1][sense(map, pos, directions[(directions.index(dir) - 1) % 4])] = 1
  vector[2][sense(map, pos, directions[(directions.index(dir) + 1) % 4])] = 1
  return vector

map = np.array(baseMap, copy=True)
pos = (3,3)
energy = 300
dir = random.choice(directions)
grabbed = False
win = False
i = 0
while energy > 0:
  clear()
  print('Energia: ', energy)
  print_state(map,pos,dir, grabbed)
  # command = input('l, r, b, f, g, v: ')
  vector = senseVector(map, pos, dir)
  
  command = mapped_movements[infer(vector)]
  pos, dir, grabbed, win = move(map, pos, dir, command, grabbed, win)
  energy -= 1
  if map[pos] == 2:
    clear()
    print('Energia: ', energy)
    print_state(map,pos,dir, grabbed)
    print('Morreu')
    break
  if win or grabbed:
    print('Venceu')
    break
  i += 1
  # time.sleep(0.01)
