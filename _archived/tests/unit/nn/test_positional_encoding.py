"""Unit tests for sinusoidal positional encodings."""

import torch

from spacenit.arch.positional_encoding import (
    cyclic_month_table,
    sinusoidal_1d,
    sinusoidal_2d,
    sinusoidal_2d_with_gsd,
)


def test_sinusoidal_1d() -> None:
    """Test that the 1-D sinusoidal position encoding is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [0.8415, 0.5332, 0.3110, 0.1769, 0.5403, 0.8460, 0.9504, 0.9842],
            [0.9093, 0.9021, 0.5911, 0.3482, -0.4161, 0.4315, 0.8066, 0.9374],
            [0.1411, 0.9933, 0.8126, 0.5085, -0.9900, -0.1160, 0.5828, 0.8610],
        ]
    )
    encoding_dim = 8
    pos = torch.tensor([0, 1, 2, 3])
    encoding = sinusoidal_1d(pos, encoding_dim)
    assert encoding.shape == (4, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_sinusoidal_2d() -> None:
    """Test that the 2-D sinusoidal position encoding is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.0000, 0.0000, 1.0000, 1.0000],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.8415, 0.3110, 0.5403, 0.9504],
            [0.0000, 0.0000, 1.0000, 1.0000, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.8415, 0.3110, 0.5403, 0.9504, 0.9093, 0.5911, -0.4161, 0.8066],
            [0.9093, 0.5911, -0.4161, 0.8066, 0.9093, 0.5911, -0.4161, 0.8066],
        ]
    )
    encoding_dim = 8
    grid = torch.tensor(
        [
            [
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            ],
        ]
    )
    encoding = sinusoidal_2d(grid, encoding_dim)
    assert encoding.shape == (18, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_sinusoidal_2d_with_gsd() -> None:
    """Test that the 2-D sinusoidal position encoding with GSD is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [
                [0.0000, 1.0000, 0.0000, 1.0000],
                [0.9093, -0.4161, 0.0000, 1.0000],
                [0.0000, 1.0000, 0.9093, -0.4161],
                [0.9093, -0.4161, 0.9093, -0.4161],
            ],
            [
                [0.0000, 1.0000, 0.0000, 1.0000],
                [0.9093, -0.4161, 0.0000, 1.0000],
                [0.0000, 1.0000, 0.9093, -0.4161],
                [0.9093, -0.4161, 0.9093, -0.4161],
            ],
        ]
    )
    encoding_dim = 4
    grid_size = 2
    pixel_pitch = torch.tensor([2.0, 2.0])
    device = torch.device("cpu")
    encoding = sinusoidal_2d_with_gsd(grid_size, pixel_pitch, encoding_dim, device)
    assert encoding.shape == (2, grid_size * grid_size, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)


def test_cyclic_month_table() -> None:
    """Test that the cyclic month encoding table is correct."""
    atol = 1e-4
    rtol = 1e-4
    expected_output = torch.tensor(
        [
            [0.0000e00, 1.0000e00],
            [5.0000e-01, 8.6603e-01],
            [8.6603e-01, 5.0000e-01],
            [1.0000e00, -4.3711e-08],
            [8.6603e-01, -5.0000e-01],
            [5.0000e-01, -8.6603e-01],
            [-8.7423e-08, -1.0000e00],
            [-5.0000e-01, -8.6603e-01],
            [-8.6603e-01, -5.0000e-01],
            [-1.0000e00, 1.1925e-08],
            [-8.6603e-01, 5.0000e-01],
            [-5.0000e-01, 8.6603e-01],
        ]
    )
    encoding_dim = 2
    encoding = cyclic_month_table(encoding_dim)
    assert encoding.shape == (12, encoding_dim)
    assert torch.allclose(encoding, expected_output, atol=atol, rtol=rtol)
