
import numpy as np
import random
import time
import neat
import os
import pickle
import visualize

 # feed_forward            = True change to False with you want a recur
 # 0 a 6 + pegou o ouro

# 0 = vazio, 1 = atencao, 2 = morte, 3 = brilho, 4 = ouro, 5 = entrada/saida

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
mapped_movements_2 = ['l','r','b','f','g','v']
# Max distante from the gold
max_dist = 14
max_dist_to_start = 12
initial_pos = (3,3)
goal_pos = (7,7)

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

def check_death(map, pos):
  if map[pos] == 2:
    return True
  return False

def sense(map, pos, dir):
  shape = map.shape
  dir = dir_dic[dir]
  dest = (pos[0] + dir[0], pos[1] + dir[1])

  if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
    return 6
  return map[dest]

def senseVector( map, pos, dir):
  vector = np.zeros((3, 7), dtype=np.int32)
  vector[0][sense(map, pos, dir)] = 1
  vector[1][sense(map, pos, directions[(directions.index(dir) - 1) % 4])] = 1
  vector[2][sense(map, pos, directions[(directions.index(dir) + 1) % 4])] = 1
  return vector

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

def infer(vecInpSens: np.int32) -> int:
    return 3     

def calculate_fitness(energy, pos, grabbed, win, reached):
  fitness = 0
  # if reached:
  #   return 100 - (abs(pos[0] - initial_pos[0]) + abs(pos[1] - initial_pos[1])) + energy / 50
  # if grabbed:
  #   return 300 - (abs(pos[0] - initial_pos[0]) + abs(pos[1] - initial_pos[1])) + energy / 50
  # elif win:
  #   return 100000
  # return max_dist - (abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])) + energy / 50
  if reached:
    fitness += 100
  if grabbed:
    fitness += 100 + max_dist_to_start - (abs(pos[0] - initial_pos[0]) + abs(pos[1] - initial_pos[1]))
  else:
    fitness += max_dist - (abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1]))
  if win:
    fitness += 100000
  
  return fitness

def train_ai(genome, config):
  net = neat.nn.RecurrentNetwork.create(genome, config)
  map = np.array(baseMap, copy=True)
  pos = (3,3)
  energy = 500
  dir = random.choice(directions)
  win, grabbed = False, False
  hist = []
  stagnation = 0
  last_commands = []
  while energy > 0 and not win and stagnation < 5:
    energy -= 3
    vector = np.append(senseVector(map, pos, dir).reshape(1, 21), grabbed).reshape(1, 22)
    output = net.activate(vector[0])
    command = mapped_movements_2[output.index(max(output))]
    # last_commands.append(command)
    # if len(last_commands) > 4: last_commands.pop(0)
    # if command != 'f' and last_commands.count(command) == 4: return 0
    pos, dir, grabbed, win = move(map, pos, dir, command, grabbed, win)
    # if (grabbed or pos == goal_pos):
    if pos in hist: stagnation += 1
    hist.append(pos)
    if (len(hist) > 20): hist.pop(0)
    # if stagnation > 50: return 0
    # if ((not grabbed or pos == goal_pos) and command == 'v'): errors += 1
    # if ((grabbed and command == 'g') or (grabbed and pos != goal_pos)): errors += 1
    if check_death(map, pos): return 0
  return calculate_fitness(energy, pos, grabbed, win, pos == goal_pos) + energy / 50

def run_neat(config):
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-199')
    # p.config.no_fitness_termination = True
    p.config.fitness_criterion = 'mean'
    p.config.fitness_threshold = 50
    # p.config.activation_mutate_rate = 2.0
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    pe = neat.ParallelEvaluator(15, train_ai)
    winner = p.run(pe.evaluate, 1000) # run for 50 generations
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

def test_best_network(config):
  with open("best.pickle", "rb") as f:
    winner = pickle.load(f)
  winner_net = neat.nn.RecurrentNetwork.create(winner, config)
  map = np.array(baseMap, copy=True)
  pos = (3,3)
  energy = 500
  dir = random.choice(directions)
  grabbed = False
  win = False
  hist = []
  stagnation = 0
  errors = 0
  resetable_stagnation = True
  last_commands = []
  while energy > 0 and not win  and stagnation < 20:
    energy -= 3
    vector = np.append(senseVector(map, pos, dir).reshape(1, 21), grabbed).reshape(1, 22)
    output = winner_net.activate(vector[0])
    command = mapped_movements_2[output.index(max(output))]
    pos, dir, grabbed, win = move(map, pos, dir, command, grabbed, win)
    os.system('cls')
    # print(output)
    print(command)
    if pos in hist: stagnation += 1
    hist.append(pos)
    if (len(hist) > 20): hist.pop(0)
    if command != 'f' and last_commands.count(command) == 4: return 0
    # if stagnation > 50: return 0
    print('fitness: ', calculate_fitness(energy, pos, grabbed, win, pos == goal_pos)) 
    print('Energia: ', energy)
    print('Grabbed: ', grabbed)
    print_state(map,pos,dir, grabbed)
    time.sleep(0.05)

    if map[pos] == 2:
      break

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # run_neat(config)
    test_best_network(config)