import random

import numpy as np


class HopfieldNet:
    def __init__(self, size: int, state: list[int] = [], seed: int = 0) -> None:
        self.size = size
        self.state = state
        self.weights = np.zeros((size, size))
        self.seed = seed  # change this to a random seed for expriments
        self.rng = np.random.default_rng(seed)

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
            previous_state = self.state.copy()
            update_order = random.sample(
                range(self.size), self.size
            )  # returns shuffled indices
            for i in update_order:
                self._update_neuron(i)
            steps += 1

        return steps
