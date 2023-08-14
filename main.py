
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


def avaliate_game(reached, grabbed, win, dead, steppedOnFlash, reachedExit, dumbness, post_grab_survive):
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
        fitness += 1000
    if grabbed and dead:
        fitness -= 10

    fitness -= dumbness
    # fitness += post_grab_survive * 0.5
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

    def infer(vecInpSens: np.int32, grabbed) -> int:
        vector = np.append(simplify_vector(vecInpSens).reshape(
            1, 21), 1 if grabbed else 0).reshape(1, 22)[0]
        output = net.activate(vector)
        return output
    return game(infer, ['f', 'l', 'r'], False, avaliate_game)


def run_neat(config):
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-630')
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6388')
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4895')
    # p.config.no_fitness_termination = True
    p.config.fitness_criterion = 'mean'
    p.config.fitness_threshold = 800
    # p.config.activation_mutate_rate = 2.0
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    pe = neat.ParallelEvaluator(15, train_ai)
    winner = p.run(pe.evaluate, 100000)  # run for X generations
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    def infer(vecInpSens: np.int32, grabbed) -> int:
        vector = np.append(simplify_vector(vecInpSens).reshape(
            1, 21), 1 if grabbed else 0).reshape(1, 22)[0]
        output = winner_net.activate(vector)
        return output
    game(infer, ['f', 'l', 'r'], True, avaliate_game)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    # test_best_network(config)
