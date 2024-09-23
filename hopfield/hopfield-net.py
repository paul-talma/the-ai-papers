# TODO: visualize (small) networks?
# TODO: visualize performance across range of parameters
# TODO: stability as fixed point


# imports
import numpy as np
from tqdm import trange
from tqdm import trange  # displays nice progress bar

# constants
n_neurons = 100  # network size
n_epochs = 5000  # epochs to stabilize

seed = 0
rng = np.random.default_rng()


def generate_memories(n_neurons, n_memories):
    """
    randomly generate n_memories memories for the net to remember

    we make sure that there are no duplicates

    params:
        n_neurons: length of memory vector
        n_memories: number of memories to generate
    return:
        memories: list of memories, each of which is a length n_neurons
        vector of activations
    """
    # we use set to test for efficient membership testing
    memories = []
    for _ in range(n_memories):
        while True:
            mem = rng.integers(low=0, high=2, size=n_neurons)
            mem = list(mem)
            if mem not in memories:
                break
        memories.append(mem)

    # convert set to list for ease of indexing later
    return memories


# could make this more efficient by looping over i and j > i,
# since the matrix is symmetric
# but over smallish matrices this won't matter
def compute_synapses(memories):
    """
    Given a list of memories to encode, computes the weight
    matrix (synapses) of the network according to Hopfield's
    eq. [2].

    loop over each connection T_ij and each memory
        if i and j have the same value in that memory, increment
        the strength of their connection
        else, decrement

    params:
        memories: list[list[int]], contains vectors of memories to encode

    return:
        synapses: np array of connection weights
    """
    n_neurons = len(memories[0])
    synapses = np.zeros((n_neurons, n_neurons))
    for i, row in enumerate(synapses):
        for j, col in enumerate(row):
            if i != j:  # T_ii = 0
                sum = 0
                for m in memories:
                    # this is the formula in Hopfield's paper, which amounts
                    # to doing: if m[i] == m[j], sum += 1, else sum -= 1
                    sum += (2 * m[i] - 1) * (2 * m[j] - 1)
                synapses[i, j] = sum

    return synapses


def update_neuron(i, synapses, activations):
    """
    Set the value of neuron i as a function of the activations of

    Sum weighted activation values of neurons in network.
    If sum is >= 0, set the activation of neuron i to 1.
    Else, set it to 0.

    Note that synapses[i][i] = 0, so the activation of i is automatically
    discounted.

    params:
        i: int, neuron to update
        synapses: np array, table of connection weights
        activations: list[int], activation values of each neuron in network

    return:
        activations: list[int], updated state of network
    """
    x_i = 0
    for j, v_j in enumerate(activations):
        x_i += synapses[i][j] * v_j
    if x_i >= 0:
        activations[i] = 1
    else:
        activations[i] = 0

    return activations


def retrieve(activations):
    """
    compute resting state of network given input vector of activations.

    do n_epochs times:
        choose neuron uniformly at random
        update value of neuron based on activities of other neurons and
        connection weights

    params:
        activations: list[int], activation values of each neuron in network

    return:
        activation: vector of neuron activations
    """
    for _ in range(n_epochs):
        choice = rng.integers(n_neurons)
        activations = update_neuron(choice, synapses, activations)
    return activations


def test_convergence():
    pass


# TODO: add parameter governing how many bits are corrupted
def test_retrieval(memories, n_tests):
    """
    Tests network's retrieval capabilities.

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
    for _ in trange(n_tests):
        # choose memory for input
        memory_choice = rng.integers(n_memories)
        # have to do the following to make sure input and mem are distinct objects
        input = [i for i in memories[memory_choice]]

        # corrupt random bit in input
        bit_choice = rng.integers(n_neurons)
        input[bit_choice] = 1 - input[bit_choice]

        output = retrieve(input)
        successes += output == memories[memory_choice]

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
    synapses = compute_synapses(memories)

    # test how long net takes to settle for different values of n_neurons

    # test quality of retrieval as n_neurons, n_memories change
    test_retrieval(memories, 100)
