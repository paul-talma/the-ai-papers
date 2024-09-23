import random

import numpy as np
from tqdm import trange

seed = 0


class HopfieldNet:
    def __init__(self, size: int, state: list[int] = []) -> None:
        self.size = size
        self.state = state
        self.weights = np.zeros((size, size))
        self.rng = np.random.default_rng(seed)  # change to random seed for experiments

    def train(self, memories: list[list[int]]) -> None:
        """
        Compute connection weights according to Hopfield's equation [2].

        params:
            memories: list of vectors representing memories to encode
        """
        for i, row in enumerate(self.weights):
            for j, col in enumerate(row):
                if j > i:  # weights are symmetric so we only compute upper triangle
                    sum = 0
                    for m in memories:
                        sum += (2 * m[i] - 1) * (2 * m[j] - 1)
                    self.weights[i, j] = self.weights[j, i] = sum

    def clear_weights(self):
        self.weights = np.zeros((self.size, self.size))

    def _update_neuron(self, i: int) -> None:
        """
        Update the value of neuron i as a function of the activations of other
        neurons and connection weights.

        Note that self.weights[i][i] = 0, so that the activation of i is
        automatically discounted.

        params:
            i: neuron to update
        """
        x_i = 0
        for j, v_j in enumerate(self.state):
            x_i += self.weights[i][j] * v_j
        if x_i >= 0:
            self.state[i] = 1
        else:
            self.state[i] = 0

    def process_input(self, input: list[int]) -> int:
        """
        Compute resting state of network given input vector of activations.

        Compare state of network before and after stochastically updating each
        neuron.
        If no change is observed, we have reached equilibrium.

        params:
            input: vector of neuron activation

        return:
            steps: complete update cycles taken to reach equilibrium
        """
        self.state = input.copy()
        previous_state = None

        steps = 0
        while self.state != previous_state:
            if steps > 999:
                break
            previous_state = self.state.copy()
            update_order = random.sample(
                range(self.size), self.size
            )  # returns shuffled indices
            for i in update_order:
                self._update_neuron(i)
            steps += 1

        return steps

    def generate_memories(self, n_memories: int) -> list[list[int]]:
        """
        Randomly generate n_memories distinct memories for the net to remember.

        params:
            n_memories: number of memories to generate.

        return:
            memories: list of memory vectors to remember.
        """
        memories = []
        for _ in range(n_memories):
            while True:
                mem = self.rng.integers(low=0, high=2, size=self.size)
                mem = list(mem)
                if mem not in memories:
                    break

            memories.append(mem)

        return memories

    def corrupt_input(self, input: list[int], n: int = 1) -> list[int]:
        bit_choices = random.sample(range(self.size), n)
        for b in bit_choices:
            input[b] = 1 - input[b]

        return input


def gen_num_memories(net_size: int = 100) -> list[int]:
    """
    Generate a list of numbers of memories to encode in the network as a
    function of network size.
    """
    base = int(net_size * 0.08)
    step = int(net_size * 0.02)
    return [base + n * step for n in range(5)]


def test_recall(net_size: int = 100, n_tests: int = 100, corrupt_bits: int = 1) -> None:
    """
    Compute network performance at recall over different number of memories.

    Generate between 0.08 * net_size and 0.18 * net_size memories for the net
    to remember.
    Do n_tests times:
        choose memory at random
        corrupt corrupt_bits bits of chosen memory
        run network on corrupt memory
        compare output to uncorrupted memory
        update running average of successful recalls

    params:
        net_size: size of network
        n_tests: number of memories to sample and corrupt
        corrupt_bits: number of bits to corrupt
    """
    net = HopfieldNet(size=net_size)

    print(f"Network size: {net_size}")
    print(f"Bits to corrupt: {corrupt_bits}\n")

    for n_memories in gen_num_memories(net.size):
        memories = net.generate_memories(n_memories)
        net.train(memories)

        success_rate = 0

        for t in trange(1, n_tests + 1):
            # choose memory to corrupt
            memory_choice = net.rng.integers(n_memories)
            input = memories[memory_choice].copy()

            # corrupt memory
            input = net.corrupt_input(input, corrupt_bits)

            # run network
            net.process_input(input)

            # record outcome
            success = net.state == memories[memory_choice]
            success_rate += (success - success_rate) / t  # incremental average

        print(f"Number of memories encoded: {n_memories}")
        print(f"Success rate (avg over {n_tests} tests): {success_rate:.2f}\n")


if __name__ == "__main__":
    net_size = 100
    n_tests = 200
    corrupt_bits = 1
    test_recall(net_size=net_size, n_tests=n_tests, corrupt_bits=corrupt_bits)
