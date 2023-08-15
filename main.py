
import numpy as np
import random
import time
import neat
import os
import pickle
from game import game


def hasFlash(vecInpSens: np.int32, i: int) -> bool:
    return vecInpSens[i][3] == 1


def hasGoal(vecInpSens: np.int32, i: int) -> bool:
    return vecInpSens[i][4] == 1


def hasObstacle(vecInpSens: np.int32, i: int) -> bool:
    return vecInpSens[i][2] == 1


def avaliate_game(reached, grabbed, win, dead, steppedOnFlash, reachedExit, dumbness):
    fitness = 0
    if steppedOnFlash:
        fitness += 5
    if reached:
        fitness += 10
    if grabbed:
        fitness += 20
    if reachedExit:
        fitness += 30
    if win:
        fitness += 100
    if dead:
        fitness -= 10

    fitness -= dumbness
    return fitness


def simplify_vector(vector):
    output = []
    for i in range(len(vector)):
        simplified_vector = np.zeros(5)
        if (vector[i][0] == 1 or vector[i][1] == 1 or vector[i][7] == 1 or vector[i][10] == 1):
            simplified_vector[0] = 1
        if (vector[i][2] == 1 or vector[i][6] == 1 or vector[i][12] == 1):
            simplified_vector[1] = 1
        if (vector[i][3] == 1 or vector[i][8] == 1 or vector[i][9] == 1 or vector[i][11] == 1):
            simplified_vector[2] = 1
        if (vector[i][4] == 1):
            simplified_vector[3] = 1
        if (vector[i][5] == 1):
            simplified_vector[4] = 1
        output.append(simplified_vector)
    return np.array(output)


def train_ai(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    def infer(vecInpSens: np.int32, onGoal, grabbed, onExit) -> int:
        vector = simplify_vector(vecInpSens).reshape(1, 15)[0]
        vector = np.append(vector, 1 if onGoal else 0)
        vector = np.append(vector, 1 if grabbed else 0)
        vector = np.append(vector, 1 if onExit else 0)
        # print(vector)
        output = net.activate(vector)
        return output
    return game(infer, ['f', 'l', 'r'], False, avaliate_game, 'baseMap')


def run_neat(config, n_generations, checkpoint=None, fitness_threashhold=800):
    p = neat.Population(config) if checkpoint == None else neat.Checkpointer.restore_checkpoint(
        f"neat-checkpoint-{checkpoint}")
    p.config.fitness_criterion = 'mean'
    p.config.fitness_threshold = fitness_threashhold
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    pe = neat.ParallelEvaluator(15, train_ai)
    winner = p.run(pe.evaluate, n_generations)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    def infer(vecInpSens: np.int32, onGoal, grabbed, onExit) -> int:
        vector = simplify_vector(vecInpSens).reshape(1, 15)[0]
        vector = np.append(vector, 1 if onGoal else 0)
        vector = np.append(vector, 1 if grabbed else 0)
        vector = np.append(vector, 1 if onExit else 0)
        output = winner_net.activate(vector)
        return output
    game(infer, ['f', 'l', 'r'], True, avaliate_game, 'baseMap')

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    choice = input("Treinar(1), salvar(2) ou testar(3): ")
    # choice = 3
    checkpoint = 1959
    populationSize = 200
    if choice == "1":
        run_neat(config, populationSize, avaliate_game, checkpoint)
    elif choice == "2":
        checkpoint = input("Checkpoint: ")
        run_neat(config, 1, checkpoint)
    elif choice == "3":
        test_best_network(config)

