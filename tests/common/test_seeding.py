import random

import numpy as np
import torch

from curriculum.common.seeding import set_seed


def test_same_seed_reproduces_all_rngs():
    gen_a = set_seed(123)
    torch_a = torch.randn(4, 4)
    legacy_a = np.random.rand(8)
    gen_draw_a = gen_a.standard_normal(8)
    py_a = random.random()

    gen_b = set_seed(123)
    torch_b = torch.randn(4, 4)
    legacy_b = np.random.rand(8)
    gen_draw_b = gen_b.standard_normal(8)
    py_b = random.random()

    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(legacy_a, legacy_b)
    assert np.array_equal(gen_draw_a, gen_draw_b)
    assert py_a == py_b


def test_different_seed_differs():
    gen_a = set_seed(0)
    torch_a = torch.randn(4, 4)
    draw_a = gen_a.standard_normal(8)

    gen_b = set_seed(1)
    torch_b = torch.randn(4, 4)
    draw_b = gen_b.standard_normal(8)

    assert not torch.equal(torch_a, torch_b)
    assert not np.array_equal(draw_a, draw_b)


def test_returns_pcg64_generator_seeded_deterministically():
    gen_a = set_seed(7)
    gen_b = set_seed(7)
    assert isinstance(gen_a, np.random.Generator)
    assert np.array_equal(gen_a.integers(0, 1000, 16), gen_b.integers(0, 1000, 16))
