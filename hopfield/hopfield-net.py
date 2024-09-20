# TODO: visualize (small) networks?
# TODO: visualize performance across range of parameters


# imports
import random

import numpy as np

# constants
n_neurons = 7  # network size
n_memories = 5  # number of memories to encode
n_epochs = 1  # epochs to stabilize

seed = 0
rng = np.random.default_rng()


def generate_synapses(memories):
    synapses = np.zeros((n_neurons, n_neurons))
    for i, row in enumerate(synapses):
        for j, col in enumerate(row):
            if i != j:
                sum = 0
                for m in memories:
                    sum += (2 * m[i] - 1) * (2 * m[j] - 1)
                synapses[i, j] = sum

    return synapses


def generate_memories():
    memories = []
    for _ in range(n_memories):
        mem = rng.integers(low=0, high=2, size=n_neurons)
        memories.append(mem)

    return memories


def update_neuron(i, synapses, activations):
    x_i = 0
    for j, v_j in enumerate(activations):
        x_i += synapses[i][j] * v_j
    if x_i >= 0:
        activations[i] = 1
    else:
        activations[i] = 0

    return activations


def retrieve(activations):
    for _ in range(n_epochs):
        choice = rng.integers(n_neurons)
        activations = update_neuron(choice, synapses, activations)
    return activations


def test_retrieval(memories, n_tests):
    """
    do n_tests times:
        select a memory at random
        corrupt random bit
        feed to network
        run net for n_epochs steps
        compare output with target

    print:
        % correct recall
        model parameters (epochs, network size, number of memories)
    """
    successes = 0
    for _ in range(n_tests):
        # choose memory for input
        memory_choice = rng.integers(n_memories)
        input = memories[memory_choice]

        # corrupt random bit in input
        bit_choice = rng.integers(n_neurons)
        input[bit_choice] = 1 - input[bit_choice]

        output = retrieve(input)
        successes += all(output == memories[memory_choice])

    print(f"Correct retrieval: {successes/n_tests}")
    print("Parameters: ")
    print(f"\tstabilization epochs: {n_epochs}")
    print(f"\tnetwork size: {n_neurons}")
    print(f"\tnumber of memories: {n_memories}")


if __name__ == "__main__":
    # initialize memories
    memories = generate_memories()

    # display first 5 memories
    print(f"First {min(n_memories, 5)} memories:")
    for i in range(min(n_memories, 5)):
        print(f"{i+1}: ", memories[i])
    print()

    # compute synaptic weights
    synapses = generate_synapses(memories)

    test_retrieval(memories, 100)

    # # feed input to net and compare with output
    # memory_choice = rng.integers(n_memories)
    # input = memories[memory_choice]
    #
    # # corrupt random bit in input
    # # TODO: corrupt n random bits
    # bit_choice = rng.integers(n_neurons)
    # input[bit_choice] = 1 - input[bit_choice]
    # print(f"corrupted input: (from memory {memory_choice + 1})")
    # print(input)
    # c = " "
    # print(c + 2 * bit_choice * c + "^\n")
    #
    # output = retrieve(input)
    # print("retrieved memory")
    # print(output, "\n")
    #
    # # correct retrieval?
    # print(f"Correct retrieval?: {all(output == memories[memory_choice])}.")
