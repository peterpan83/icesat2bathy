import torch
import torch.nn as nn

def fps_1d(points, num_samples):
    """
    Since pointnet2_utils.furthest_point_sample and
    pointnet2_utils.gather_operation does no work for 1D points based on testing,
    this pure python function is implemented.

    Args:
        points: Tensor of shape (B, N, 1) where B is batch size, N is number of points
        num_samples: Number of points to sample

    Returns:
        sampled_indices: Tensor of shape (B, num_samples) containing indices of sampled points
        sampled_points: Tensor of shape (B, num_samples, 1) containing the sampled points
    """
    B, N, _ = points.shape
    device = points.device

    # Initialize
    distances = torch.full((B, N), float('inf'), device=device)
    sampled_indices = torch.zeros((B, num_samples), dtype=torch.long, device=device)

    # Randomly select first point for each batch
    first_indices = torch.randint(0, N, (B,), device=device)
    sampled_indices[:, 0] = first_indices

    # Get first points and compute initial distances
    batch_idx = torch.arange(B, device=device)
    first_points = points[batch_idx, first_indices, 0]  # Shape: (B,)
    distances = torch.abs(points[:, :, 0] - first_points.unsqueeze(1))  # Broadcasting

    # Iteratively select farthest points
    for i in range(1, num_samples):
        # Find farthest points
        farthest_indices = torch.argmax(distances, dim=1)
        sampled_indices[:, i] = farthest_indices

        # Get new points and compute distances
        new_points = points[batch_idx, farthest_indices, 0]  # Shape: (B,)
        new_distances = torch.abs(points[:, :, 0] - new_points.unsqueeze(1))

        # Update minimum distances
        distances = torch.minimum(distances, new_distances)

    # Gather sampled points
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_samples)
    sampled_points = points[batch_indices, sampled_indices]

    return sampled_indices, sampled_points



class KNN(nn.Module):
    def __init__(self, k, transpose_mode=True):
        super(KNN, self).__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, coords, center_coords=None):
        """
        coords: torch.Tensor of shape (B, N, D)
        center_coords: torch.Tensor of shape (B, M, D) if provided,
                       or None to default to self-knn (coords vs coords)
        Returns:
            dist: torch.Tensor of shape (B, M, k)
            idx: torch.Tensor of shape (B, M, k) — indices in `coords`
        """
        if center_coords is None:
            center_coords = coords

        if coords.is_cuda or center_coords.is_cuda:
            raise RuntimeError("This implementation is CPU-only.")

        B, N, D = coords.shape
        M = center_coords.shape[1]

        # Compute squared distances: (B, M, N)
        coords_sq = torch.sum(coords ** 2, dim=-1, keepdim=True)     # (B, N, 1)
        center_sq = torch.sum(center_coords ** 2, dim=-1, keepdim=True)  # (B, M, 1)

        # dist[b, m, n] = ||center[m] - coords[n]||^2
        dist = (
            center_sq + coords_sq.transpose(1, 2)
            - 2 * torch.bmm(center_coords, coords.transpose(1, 2))
        )

        # Get k smallest distances (sorted)
        dist, idx = torch.topk(dist, self.k, dim=-1, largest=False, sorted=True)
        dist = dist.clamp(min=1e-10).sqrt()

        return dist, idx




