
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
    clear()
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

# if __name__ == '__main__':
#   map = np.array(baseMap, copy=True)
#   pos = (3,3)
#   energy = 300
#   dir = random.choice(directions)
#   grabbed = False
#   win = False
#   i = 0
#   while energy > 0:
#     os.system('cls')
#     print('Energia: ', energy)
#     print_state(map,pos,dir, grabbed)
#     # command = input('l, r, b, f, g, v: ')
#     vector = senseVector(map, pos, dir)
    
#     command = mapped_movements[infer(vector)]
#     pos, dir, grabbed, win = move(map, pos, dir, command, grabbed, win)
#     energy -= 1
#     if map[pos] == 2:
#       os.system('cls')
#       print('Energia: ', energy)
#       print_state(map,pos,dir, grabbed)
#       print('Morreu')
#       break
#     if win or grabbed:
#       print('Venceu')
#       break
#     i += 1
#     # time.sleep(0.01)

def train_ai(genome, config):
  net = neat.nn.FeedForwardNetwork.create(genome, config)
  map = np.array(baseMap, copy=True)
  pos = (3,3)
  energy = 500
  dir = random.choice(directions)
  grabbed = False
  win = False
  i = 0
  while energy > 0:
    energy -= 3
    vector = np.append(senseVector(map, pos, dir).reshape(1, 21), grabbed).reshape(1, 22)
    output = net.activate(vector[0])
    command = mapped_movements_2[output.index(max(output))]
    pos, dir, grabbed, win = move(map, pos, dir, command, grabbed, win)
    os.system('cls')
    print(output)
    print(command)
    print('Energia: ', energy)
    print_state(map,pos,dir, grabbed)
    if map[pos] == 2:
      break
    if win or grabbed:
      break
    i += 1
  genome.fitness = i

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
      genome.fitness = 500
      train_ai(genome, config)


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 1) # run for 50 generations
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    # test_best_network(config)