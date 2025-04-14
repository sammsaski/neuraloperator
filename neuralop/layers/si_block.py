import torch
import torch.nn as nn
import torch.nn.functional as F


class SIBlocks(nn.Module):
	"""

	Parameters
	----------
	in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
	num_knots : int
		the number of knots to use when estimating the kernel function with a spline
		
	Other Parameters
	----------------	
	"""
	
	def __init__(
		self,
		in_channels,
		out_channels,
		num_knots,
		radius_cutoff=0.2,
		max_neighbors=32,
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_knots = num_knots
		self.radius_cutoff = radius_cutoff
		self.max_neighbors = max_neighbors

		self.phi_net = nn.Sequential(
			nn.Linear(2 * in_channels, 32),
			nn.ReLU(),
			# nn.Linear(128, 128),
			# nn.ReLU(),
			nn.Linear(32, out_channels),

			# WHAT WE'VE BEEN USING ORIGINAL
			# nn.Linear(2 * in_channels, out_channels),
			# nn.ReLU(),
			# nn.Linear(out_channels, 1)
		)

		self.h_net = nn.Sequential(
			# nn.Linear(in_channels, out_channels),
			# nn.ReLU(),
			nn.Linear(in_channels, 32),
			nn.ReLU(),
			nn.Linear(32, out_channels),
			# nn.Linear(out_channels, out_channels),
			# nn.ReLU(),
			# nn.Linear(out_channels, 1),
			nn.Softplus() # ensures h(x) > 0
		)

		# randomly initialize spline weights
		self.S_m = nn.Parameter(torch.randn(num_knots))

		# evenly space knots
		knots = torch.linspace(0, 1, steps=num_knots)
		self.register_buffer("knot_points", knots)

	def radial_spline(self, r):
		"""
		Spline basis (linear) interpolation across radial bins.
		"""
		r = torch.clamp(r, 0.0, 1.0)

		idx = torch.floor(r * (self.num_knots - 1)).long()
		idx = torch.clamp(idx, 0, self.num_knots - 2)

		# get knot values
		w_k = self.S_m[idx]
		w_k1 = self.S_m[idx + 1]

		# knot locations for interpolation
		t_k = self.knot_points[idx]
		t_k1 = self.knot_points[idx + 1]

		# linear interpolation
		weight_right = (r - t_k) / (t_k1 - t_k + 1e-8)
		weight_left = 1.0 - weight_right
		return weight_left * w_k + weight_right * w_k1
	
	def forward(self, x):
		"""
		"""
		B, N, C = x.shape

		# 1. Generate 2D grid of coordinates (normalized)
		H = W = int(N**0.5)
		coords = torch.stack(torch.meshgrid(
			torch.linspace(0, 1, H, device=x.device),
			torch.linspace(0, 1, W, device=x.device),
			indexing='ij'
		), dim=-1).reshape(1, N, 2).repeat(B, 1, 1)  # (B, N, 2)

		# 2. Pairwise distances between all (x_i, y_j)
		x_q = coords.unsqueeze(2)        # (B, N, 1, 2)
		y_k = coords.unsqueeze(1)        # (B, 1, N, 2)
		pairwise_diff = x_q - y_k        # (B, N, N, 2)
		dist = torch.norm(pairwise_diff, dim=-1)  # (B, N, N)

		mask = (dist <= self.radius_cutoff).float()

		# 3. Compute h(x_i): local scaling
		h = self.h_net(coords).squeeze(-1)  # (B, N)
		h = h.unsqueeze(-1)                # (B, N, 1)
		r_scaled = dist / (h + 1e-6)       # (B, N, N), normalized distances

		# 4. Evaluate spline kernel ψ(r)
		psi = self.radial_spline(r_scaled) * mask  # (B, N, N)

		# 5. Compute φ(x_i, y_j)
		# xy_pairs = torch.cat([
		# 	x_q.expand(-1, -1, N, -1),  # (B, N, N, 2)
		# 	y_k.expand(-1, N, -1, -1)   # (B, N, N, 2)
		# ], dim=-1)                      # (B, N, N, 4)
		# phi = self.phi_net(xy_pairs).squeeze(-1) * mask  # (B, N, N)

		# Create pairwise inputs only for entries where mask is 1
		# (B, N, N) → (num_active, 3) where each row = (b, i, j)
		active_idx = mask.nonzero(as_tuple=False)  # (num_active, 3)

		# Extract b, i, j indices
		b_idx, i_idx, j_idx = active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]

		# Gather coords for x_i and y_j
		x_i = coords[b_idx, i_idx]  # (num_active, 2)
		y_j = coords[b_idx, j_idx]  # (num_active, 2)

		# Form [x_i, y_j] pairs as input to phi_net
		xy_input = torch.cat([x_i, y_j], dim=-1)  # (num_active, 4)

		# Evaluate φ(x_i, y_j) only on active entries
		phi_vals = self.phi_net(xy_input).squeeze(-1)  # (num_active,)

		# Scatter φ values back into full tensor
		phi = torch.zeros_like(dist)  # (B, N, N)
		phi[b_idx, i_idx, j_idx] = phi_vals

		# 6. Weight input features u(y_j) = x: shape (B, N, C)
		# u_y = x.unsqueeze(1)            # (B, 1, N, C) → broadcasted across queries
		# weights = phi * psi             # (B, N, N)
		# weights = weights.unsqueeze(-1)  # (B, N, N, 1)

		# # 7. Apply kernel: weighted sum over input points y_j
		# out = (weights * u_y).sum(dim=2) / N  # (B, N, C)


		# Kernel weights
		weights = phi * psi  # (B, N, N)
		weights = weights.unsqueeze(-1)  # (B, N, N, 1)

		# Apply kernel operator to u(y) = x
		u_y = x.unsqueeze(1)  # (B, 1, N, C)
		weighted = weights * u_y  # (B, N, N, C)

		# Normalize over contributing neighbors
		normalization = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, N, 1)
		out = weighted.sum(dim=2) / normalization  # (B, N, C)


		return out