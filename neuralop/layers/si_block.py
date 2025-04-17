import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import time


# class FastPhi(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.proj_i = nn.Linear(2, embed_dim, bias=False)
#         self.proj_j = nn.Linear(2, embed_dim, bias=False)

#     def forward(self, coords_i, coords_j):
#         # Simple bilinear interaction between input points
#         i_embed = self.proj_i(coords_i)  # (B*N*K, D)
#         j_embed = self.proj_j(coords_j)  # (B*N*K, D)
#         interaction = (i_embed * j_embed).sum(dim=-1)  # (B*N*K,)
#         return interaction

class FastPhi(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embed_dim))
        self.i_proj = nn.Linear(2, embed_dim, bias=False)
        self.j_proj = nn.Linear(2, embed_dim, bias=False)

    def forward(self, coords_i, coords_j):
        i = self.i_proj(coords_i)  # (B*N*K, D)
        j = self.j_proj(coords_j)  # (B*N*K, D)
        return F.linear(i * j, self.weight.unsqueeze(0))[:, 0]  # (B*N*K,)



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

		self.W = nn.Sequential(
			nn.Linear(in_channels, out_channels*2),
			nn.ReLU(),
			nn.Linear(out_channels*2, out_channels)
		)

		# self.phi_net = nn.Sequential(
		# 	nn.Linear(4, in_channels),
		# 	nn.ReLU(),
		# 	nn.Linear(in_channels, out_channels),

		# 	# WHAT WE'VE BEEN USING ORIGINAL
		# 	# nn.Linear(2 * in_channels, out_channels),
		# 	# nn.ReLU(),
		# 	# nn.Linear(out_channels, 1)
		# )

		self.phi_net = FastPhi(embed_dim=in_channels)

		self.h_net = nn.Sequential(
			# nn.Linear(in_channels, out_channels),
			# nn.ReLU(),
			nn.Linear(2, in_channels),
			nn.ReLU(),
			nn.Linear(in_channels, 1),
			# nn.Linear(out_channels, out_channels),
			# nn.ReLU(),
			# nn.Linear(out_channels, 1),
			nn.Softplus() # ensures h(x) > 0
		)

		# randomly initialize spline weights
		# self.S_m = nn.Parameter(torch.randn(num_knots))
		self.S_m_x = nn.Parameter(torch.randn(num_knots))
		self.S_m_y = nn.Parameter(torch.randn(num_knots))

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
	
	@staticmethod
	def bspline_basis_1d(x, knots, weights):
		"""
		x: (N,) tensor of input positions
		knots: (K,) tensor of knot positions (monotonic)
		weights: (K,) tensor of learned weights (S_m)
		returns: (N,) spline interpolation of x using linear B-spline
		"""
		x = x.unsqueeze(-1)  # (N, 1)
		knots = knots.unsqueeze(0)  # (1, K)

		# Compute basis functions: linear B-spline
		dists = (x - knots).abs()
		mask = (dists < 1)  # only linear influence in [-1, 1]
		basis = (1 - dists) * mask  # (N, K)

		return (basis * weights).sum(-1)  # (N,)

	
	def forward(self, x):
		"""
		Forward pass with neighbor subsampling.
		"""
		B, N, C = x.shape
		H = W = int(N**0.5)

		coords = torch.stack(torch.meshgrid(
			torch.linspace(0, 1, H, device=x.device),
			torch.linspace(0, 1, W, device=x.device),
			indexing='ij'
		), dim=-1).reshape(1, N, 2).repeat(B, 1, 1)  # (B, N, 2)

		start = time.time()

		# Step 1: Compute pairwise distances
		x_q = coords.unsqueeze(2)  # (B, N, 1, 2)
		y_k = coords.unsqueeze(1)  # (B, 1, N, 2)
		pairwise_diff = x_q - y_k  # (B, N, N, 2)
		dist = torch.norm(pairwise_diff, dim=-1)  # (B, N, N)
		mask = (dist <= self.radius_cutoff)

		# t0 = time.time()
		# print(f'Time to compute pairwise dist: {t0 - start}')

		# Step 2: Sample K neighbors within radius
		dist_masked = dist.masked_fill(~mask, float('inf'))
		topk_dists, topk_idx = torch.topk(dist_masked, k=self.max_neighbors, dim=-1, largest=False)

		# t1 = time.time()
		# print(f'Time to sample k neighbors: {t1 - t0}')

		# Step 3: Build sample indices
		B, N, K = topk_idx.shape
		device = x.device
		batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
		point_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)

		b_idx = batch_idx.reshape(-1)  # (B*N*K,)
		i_idx = point_idx.reshape(-1)  # (B*N*K,)
		j_idx = topk_idx.reshape(-1)   # (B*N*K,)

		# Step 4: Get coords and compute r
		coords_i = coords[b_idx, i_idx]  # (B*N*K, 2)
		coords_j = coords[b_idx, j_idx]  # (B*N*K, 2)
		r = torch.norm(coords_i - coords_j, dim=-1)  # (B*N*K,)

		# Step 5: h(x_i) and scaled r
		h_i = self.h_net(coords_i).squeeze(-1)  # (B*N*K,)
		r_scaled = r / (h_i + 1e-6)

		# Step 6: Spline ψ(r)
		# start = time.time()
		# psi_vals = self.radial_spline(r_scaled)  # (B*N*K,)

		### NEW B-SPLINE BASIS IMPLEMENTATION
		relative_coords = coords_i - coords_j  # (B*N*K, 2)
		r_proj = relative_coords.norm(dim=-1)  # or use just one axis like relative_coords[:, 0]
		psi_x = SIBlocks.bspline_basis_1d(relative_coords[:, 0], self.knot_points, self.S_m_x)
		psi_y = SIBlocks.bspline_basis_1d(relative_coords[:, 1], self.knot_points, self.S_m_y)
		psi_vals = psi_x * psi_y  # separable 2D spline

		psi_vals = psi_vals / (psi_vals.abs().mean() + 1e-6) # normalize
		# t2 = time.time()
		# print(f'Time to fit spline: {t2 - start}')


		# Step 7: φ(x_i, y_j)
		# xy_input = torch.cat([coords_i, coords_j], dim=-1)  # (B*N*K, 4)
		# phi_vals = self.phi_net(xy_input).squeeze(-1)       # (B*N*K,)
		# phi_vals = phi_vals / (phi_vals.abs().mean(dim=0, keepdim=True) + 1e-6) # normalize
		phi_vals = self.phi_net(coords_i, coords_j)
		phi_vals = phi_vals / (phi_vals.abs().mean() + 1e-6)  # normalize over all

		# Ensure both are shape (B*N*K,)
		phi_vals = phi_vals.view(-1)
		psi_vals = psi_vals.view(-1)

		# Combine them properly → shape (B*N*K, 1)
		weights = (psi_vals * phi_vals).unsqueeze(-1)  # NOT unsqueeze before multiply


		# Step 8: Weights and u(y_j)
		# weights = psi_vals * phi_vals  # (B*N*K,)
		# weights = psi_vals.unsqueeze(-1) * phi_vals
		
		# u_y = x[b_idx, j_idx]          # (B*N*K, C)
		# NEW u_y ASSIGNMENT FOR SPEED UP
		x_flat = x.reshape(B * N, C)  # (B*N, C)
		gather_idx = b_idx * N + j_idx  # (B*N*K,)
		u_y = x_flat[gather_idx]  # now a flat gather


		# weighted = weights.unsqueeze(-1) * u_y  # (B*N*K, C)
		weighted = weights * u_y

		# Step 9: Scatter into output
		out = torch.zeros(B * N, C, device=device)
		idx_flat = b_idx * N + i_idx  # Flattened index for each (b, i)

		# ADDED FOR SPEEDUP
		weighted = weighted.contiguous()
		idx_flat = idx_flat.contiguous()
		### 

		out.index_add_(0, idx_flat, weighted)
		out = out.view(B, N, C)

		# Step 10: Normalize
		# normalizer = torch.zeros(B * N, device=device)
		# normalizer.index_add_(0, idx_flat, torch.ones_like(weights))
		# normalizer = normalizer.clamp(min=1.0).view(B, N, 1)

		# start = time.time()
		normalizer = torch.zeros(B * N, C, device=device)
		normalizer.index_add_(0, idx_flat, torch.ones_like(weighted))
		normalizer = normalizer.clamp(min=1.0).view(B, N, C)
		# # t4 = time.time()
		# # print(f'Normalize time: {t4 - start}')

		out = out / normalizer


		# Step 11: Pointwise path
		# start = time.time()
		pointwise_out = self.W(x)
		# t5 = time.time()
		# print(f'Time for pointwise W: {t5 - start}')

		# t3 = time.time()
		# print(f'Time for everything else: {t3 - t2}')

		# print(out.shape)
		# print(pointwise_out.shape)

		return out + pointwise_out
	
	def plot_and_save_spline(self, save_path="learned_spline.png"):
		"""
		Plot and save the learned spline ψ(r) from a SIBlocks layer.
		
		Parameters
		----------
		layer : SIBlocks
			An instance of your spline-based operator layer.
		save_path : str
			Path to save the plot image.
		"""
		with torch.no_grad():
			# r_vals = torch.linspace(0, 1, 200).to(self.S_m.device)
			r_vals = torch.linspace(0, 1, 200).to(self.S_m_x.device)
			spline_vals = self.radial_spline(r_vals)

		plt.figure(figsize=(6, 4))
		plt.plot(r_vals.cpu(), spline_vals.cpu(), label="ψ(r)")
		plt.title("Learned Spline Kernel ψ(r)")
		plt.xlabel("r (normalized distance)")
		plt.ylabel("ψ(r)")
		plt.grid(True)
		plt.legend()

		# Make directory if it doesn't exist
		os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
		plt.savefig(save_path)
		plt.close()
