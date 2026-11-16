import numpy as np
from typing import List, Tuple, Optional

class SimpleMLP:
    """
    Feedforward Neural Network implemented with pure NumPy.
    Weights and biases are easily packable/unpackable into a 1D chromosome
    for Genetic Algorithm mutation and crossover.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 4,
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        layer_dims = [input_dim] + self.hidden_dims + [output_dim]
        self.layer_shapes: List[Tuple[int, int]] = []
        for i in range(len(layer_dims) - 1):
            self.layer_shapes.append((layer_dims[i], layer_dims[i + 1]))

        if weights is not None and biases is not None:
            self.weights = [np.array(w, dtype=np.float32) for w in weights]
            self.biases = [np.array(b, dtype=np.float32) for b in biases]
        else:
            self.weights = []
            self.biases = []
            for in_d, out_d in self.layer_shapes:
                # Xavier / Glorot uniform initialization
                limit = np.sqrt(6.0 / (in_d + out_d))
                w = np.random.uniform(-limit, limit, size=(in_d, out_d)).astype(np.float32)
                b = np.zeros(out_d, dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with LeakyReLU activations on hidden layers
        and Tanh on the output layer to strictly constrain output to [-1.0, 1.0].
        """
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h.reshape(1, -1)

        for i in range(len(self.weights) - 1):
            h = np.dot(h, self.weights[i]) + self.biases[i]
            # LeakyReLU (slope 0.1)
            h = np.where(h > 0, h, h * 0.1)

        # Output layer
        out = np.dot(h, self.weights[-1]) + self.biases[-1]
        out = np.tanh(out)
        return out.flatten()

    def get_chromosome(self) -> np.ndarray:
        """Flattens all weights and biases into a 1D numpy vector."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b.flatten())
        return np.concatenate(parts).astype(np.float32)

    def set_chromosome(self, chromosome: np.ndarray):
        """Unpacks 1D vector back into weights and biases."""
        idx = 0
        new_weights = []
        new_biases = []
        for in_d, out_d in self.layer_shapes:
            w_size = in_d * out_d
            w = chromosome[idx : idx + w_size].reshape((in_d, out_d))
            idx += w_size

            b_size = out_d
            b = chromosome[idx : idx + b_size]
            idx += b_size

            new_weights.append(w)
            new_biases.append(b)

        self.weights = new_weights
        self.biases = new_biases

    def save(self, filepath: str):
        """Saves model architecture and weights to a compressed .npz file."""
        np.savez_compressed(
            filepath,
            input_dim=self.input_dim,
            hidden_dims=np.array(self.hidden_dims, dtype=np.int32),
            output_dim=self.output_dim,
            chromosome=self.get_chromosome(),
        )

    @classmethod
    def load(cls, filepath: str) -> "SimpleMLP":
        """Loads model from a compressed .npz file."""
        data = np.load(filepath)
        input_dim = int(data["input_dim"])
        hidden_dims = list(data["hidden_dims"])
        output_dim = int(data["output_dim"])
        chromosome = data["chromosome"]

        model = cls(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        model.set_chromosome(chromosome)
        return model
