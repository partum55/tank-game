import numpy as np
import tempfile
import os
from tank_env.neural_net import SimpleMLP
from tank_env.adapters import VectorObservationAdapter
from tank_env.environment import TankEnvironment

def test_mlp_forward_and_chromosome():
    mlp = SimpleMLP(input_dim=14, hidden_dims=[16, 16], output_dim=4)
    chrom = mlp.get_chromosome()
    assert isinstance(chrom, np.ndarray)
    assert chrom.ndim == 1

    new_chrom = chrom + np.random.randn(*chrom.shape) * 0.1
    mlp.set_chromosome(new_chrom)

    dummy_input = np.zeros(14, dtype=np.float32)
    output = mlp.forward(dummy_input)
    assert output.shape == (4,)
    assert np.all(output >= -1.0) and np.all(output <= 1.0)

def test_mlp_save_and_load():
    mlp = SimpleMLP(input_dim=14, hidden_dims=[16, 16], output_dim=4)
    dummy_input = np.ones(14, dtype=np.float32) * 0.5
    out1 = mlp.forward(dummy_input)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        mlp.save(tmp_path)
        loaded_mlp = SimpleMLP.load(tmp_path)
        out2 = loaded_mlp.forward(dummy_input)
        np.testing.assert_allclose(out1, out2, rtol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
