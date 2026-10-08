"""The TabPFN trust grid the phone interpolates must be well-formed and make sense."""
import json
from pathlib import Path

import numpy as np

trust = json.loads((Path(__file__).parent.parent / "app" / "data" / "trust.json").read_text())


def test_grid_shape_and_range():
    n_p, n_cos = len(trust["p"]), len(trust["cos"])
    for kingdom in ("plants", "fungi"):
        g = np.array(trust[kingdom])
        assert g.shape == (n_p, n_p, n_cos)
        assert g.min() >= 0 and g.max() <= 1
    assert 0 < trust["likely"] < trust["confident"] < 1


def test_more_certain_model_means_more_trust():
    # Averaged over the other features, a higher top-1 probability should not lower the trust.
    for kingdom in ("plants", "fungi"):
        by_p1 = np.array(trust[kingdom]).mean(axis=(1, 2))
        assert by_p1[-1] > by_p1[0]
