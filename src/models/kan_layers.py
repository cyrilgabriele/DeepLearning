import torch
import torch.nn as nn
import math


class ChebyKANLayer(nn.Module):
    """KAN layer using Chebyshev polynomial basis functions.

    Each edge (i, j) learns a function as a linear combination of
    Chebyshev polynomials T_0(x) through T_d(x). Inputs are normalized
    to [-1, 1] via tanh for numerical stability.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(
            torch.empty(out_features, in_features, degree + 1)
        )
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cheby_coeffs, mean=0.0,
                        std=1.0 / (self.in_features * (self.degree + 1)))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)

        cheby = [torch.ones_like(x_norm), x_norm]
        for n in range(2, self.degree + 1):
            cheby.append(2 * x_norm * cheby[-1] - cheby[-2])
        cheby_basis = torch.stack(cheby, dim=-1)

        kan_out = torch.einsum("bid,oid->bo", cheby_basis, self.cheby_coeffs)
        base_out = nn.functional.linear(x, self.base_weight)

        return kan_out + base_out


class FourierKANLayer(nn.Module):
    """KAN layer using Fourier series basis functions.

    Each edge (i, j) learns a function as a sum of cosine and sine terms
    with learned coefficients.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.fourier_a = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.fourier_b = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fourier_a, mean=0.0,
                        std=1.0 / (self.in_features * self.grid_size))
        nn.init.normal_(self.fourier_b, mean=0.0,
                        std=1.0 / (self.in_features * self.grid_size))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype)

        x_scaled = (torch.tanh(x) + 1) * math.pi

        x_k = x_scaled.unsqueeze(-1) * k
        cos_basis = torch.cos(x_k)
        sin_basis = torch.sin(x_k)

        kan_cos = torch.einsum("big,oig->bo", cos_basis, self.fourier_a)
        kan_sin = torch.einsum("big,oig->bo", sin_basis, self.fourier_b)

        base_out = nn.functional.linear(x, self.base_weight)

        return kan_cos + kan_sin + base_out


class BSplineKANLayer(nn.Module):
    """KAN layer using B-spline basis functions (original KAN, Liu et al. 2024).

    Each edge learns a function as a linear combination of B-spline basis
    functions on a uniform grid. This is the reference KAN architecture.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform grid on [-1, 1] extended by spline_order on each side
        num_knots = grid_size + 2 * spline_order + 1
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - spline_order * h, 1 + spline_order * h, num_knots)
        self.register_buffer("grid", grid)

        # B-spline coefficients: one per (output, input, basis function)
        num_bases = grid_size + spline_order
        self.spline_coeffs = nn.Parameter(
            torch.empty(out_features, in_features, num_bases)
        )
        # Residual linear path for training stability
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.spline_coeffs, mean=0.0,
                        std=1.0 / (self.in_features * (self.grid_size + self.spline_order)))
        nn.init.xavier_uniform_(self.base_weight)

    def _b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions using Cox-de Boor recursion.

        Args:
            x: (batch, in_features) input tensor, expected in [-1, 1].

        Returns:
            (batch, in_features, num_bases) tensor of basis function values.
        """
        grid = self.grid  # (num_knots,)
        x = x.unsqueeze(-1)  # (batch, in, 1)

        # Order 0: piecewise constant
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # (batch, in, num_knots-1)

        # Cox-de Boor recursion for orders 1..spline_order
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:-(k + 1)]
            left_den = grid[k:-1] - grid[:-(k + 1)]
            left = left_num / left_den.clamp(min=1e-8) * bases[:, :, :-1]

            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:(-k if -k != 0 else None)]
            right = right_num / right_den.clamp(min=1e-8) * bases[:, :, 1:]

            bases = left + right

        return bases  # (batch, in, num_bases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [-1, 1]
        x_norm = torch.tanh(x)

        # Compute B-spline basis values
        basis = self._b_spline_basis(x_norm)  # (batch, in, num_bases)

        # Weighted sum: (batch, in, num_bases) @ (out, in, num_bases) -> (batch, out)
        kan_out = torch.einsum("bin,oin->bo", basis, self.spline_coeffs)

        # Residual linear path
        base_out = nn.functional.linear(x, self.base_weight)

        return kan_out + base_out
