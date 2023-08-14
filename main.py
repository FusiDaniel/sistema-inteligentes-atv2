
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


def avaliate_output(output, vecInpSens, pos, last_command, isInGoal, grabbed):
    fitness = 0
    # primeiro confere flash e goal
    if (isInGoal and not grabbed):
        return 1000
        # return 1000 if output == 0 else -10
    elif (hasFlash(vecInpSens, 0) and not grabbed):
        fitness = 100 if output == 3 else -100
    elif ((hasFlash(vecInpSens, 1) or hasGoal(vecInpSens, 1)) and not grabbed):
        fitness = 100 if output == 11 else -100
    elif ((hasFlash(vecInpSens, 2) or hasGoal(vecInpSens, 2)) and not grabbed):
        fitness = 100 if output == 12 else -100

    elif (hasObstacle(vecInpSens, 0)):
        if (hasObstacle(vecInpSens, 1)):
            fitness = 2 if output in (12, 13) else -5
        elif (hasObstacle(vecInpSens, 2)):
            fitness = 2 if output in (11, 13) else -5
        else:
            if (last_command == 11):
                fitness = 2 if output in (12, 13) else -5
            elif (last_command == 12):
                fitness = 2 if output in (11, 13) else -5
            else:
                fitness = 2 if output in (11, 12, 13) else -5
    else:
        if (last_command == 11 or last_command == 12):
            fitness = 2 if output == 3 else -5
        elif (hasObstacle(vecInpSens, 1)):
            fitness = 2 if output in (3, 13) else -5
        elif (hasObstacle(vecInpSens, 2)):
            fitness = 2 if output in (3, 11) else -5
        else:
            fitness = 0 if output in (3, 11, 12) else -5
    return fitness


def simplify_vector(vector):
    output = []
    for i in range(len(vector)):
        simplified_vector = vector[i][:7]
        if (vector[i][7] == 1 or vector[i][10] == 1):
            simplified_vector[1] = 1
        if (vector[i][8] == 1 or vector[i][9] == 1 or vector[i][11] == 1):
            simplified_vector[3] = 1
        if (vector[i][6] == 1 or vector[i][12] == 1):
            simplified_vector[2] = 1
        output.append(simplified_vector)
    return np.array(output)


def train_ai(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    def infer(vecInpSens: np.int32) -> int:
        vector = simplify_vector(vecInpSens).reshape(1, 21)[0]
        output = net.activate(vector)
        return output
    return game(infer, ['f', 'l', 'r'], False, avaliate_output)


def run_neat(config):
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1297')
    # p.config.no_fitness_termination = True
    p.config.fitness_criterion = 'mean'
    p.config.fitness_threshold = 900
    # p.config.activation_mutate_rate = 2.0
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    pe = neat.ParallelEvaluator(15, train_ai)
    winner = p.run(pe.evaluate, 1000)  # run for X generations
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    def infer(vecInpSens: np.int32) -> int:
        vector = simplify_vector(vecInpSens).reshape(1, 21)[0]
        output = winner_net.activate(vector)
        return output
    game(infer, ['f', 'l', 'r'], True, avaliate_output)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    # test_best_network(config)
