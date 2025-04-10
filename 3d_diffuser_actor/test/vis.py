import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image

def load_depth_16bit(path, max_depth=10.0):
    """
    Loads a 16-bit depth PNG and rescales it to floating-point depth values.
    
    Args:
        path (str): Path to the 16-bit PNG file.
        max_depth (float): The maximum depth used during saving.
    
    Returns:
        np.ndarray: Depth image in float32, scaled to [0, max_depth].
    """
    img = Image.open(path)
    depth_uint16 = np.array(img).astype(np.uint16)
    depth = (depth_uint16.astype(np.float32) / 65535.0) * max_depth
    return depth

def visualize_depth(depth_array, cmap='viridis', normalize=True, save_path='depth.png'):
    """
    Visualizes a depth map using matplotlib.

    Args:
        depth_array (np.ndarray): 2D array of depth values.
        cmap (str): Matplotlib colormap (e.g., 'viridis', 'plasma', 'inferno').
        normalize (bool): Whether to normalize the depth for better visualization.
    """
    if normalize:
        depth_min, depth_max = np.min(depth_array), np.max(depth_array)
        print(f"Depth range: [{depth_min:.3f}, {depth_max:.3f}]")
        depth_array = (depth_array - depth_min) / (depth_max - depth_min + 1e-8)

    plt.imshow(depth_array, cmap=cmap)
    plt.colorbar(label='Normalized Depth')
    plt.title("Depth Map")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

def main():
    parser = argparse.ArgumentParser(description="Visualize a depth .npy file")
    parser.add_argument("file", type=str, help="Path to depth .npy file")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")

    args = parser.parse_args()
    assert os.path.isfile(args.file), "Depth file not found."

    depth = load_depth_16bit(args.file)

    save_path = os.path.splitext(args.file)[0] + "_visualized.png"
    print(save_path)
    visualize_depth(depth, cmap=args.cmap, normalize=not args.no_normalize, save_path=save_path)

if __name__ == "__main__":
    main()
