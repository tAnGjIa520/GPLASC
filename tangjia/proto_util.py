import torch

def farthest_point_on_sphere(points, device="cpu"):
    """
    Finds a point on the hypersphere farthest from a set of points.

    Args:
        points (torch.Tensor): Tensor of shape (N, D), where N is the number of points
                              and D is the dimension of the hypersphere.
        device (str): Device to run the computation on (default: "cpu").

    Returns:
        torch.Tensor: A tensor of shape (D,) representing the farthest point.
    """
    # Ensure points are on the unit sphere
    points = points.to(device)
    points = points / torch.norm(points, dim=1, keepdim=True)

    # Compute the mean direction vector
    mean_direction = torch.mean(points, dim=0)
    mean_direction = mean_direction / torch.norm(mean_direction)

    # Find the farthest point by flipping the mean direction
    farthest_point = -mean_direction

    return farthest_point

# Example usage
if __name__ == "__main__":
    # Generate random points on a 3D hypersphere
    torch.manual_seed(42)
    N, D = 100, 3  # 100 points in 3 dimensions
    random_points = torch.randn(N, D)
    random_points = random_points / torch.norm(random_points, dim=1, keepdim=True)

    # Find the farthest point
    farthest_point = farthest_point_on_sphere(random_points)
    print("Farthest point on the hypersphere:", farthest_point)
    print("Norm of the farthest point:", torch.norm(farthest_point))
