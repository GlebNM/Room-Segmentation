import os
import numpy as np
import argparse
import open3d as o3d
import colorsys

parser = argparse.ArgumentParser()
parser.add_argument('--room_path', type=str, required=True)
parser.add_argument('--color_alpha', type=float, default=0.5)

def generate_color_palette(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def main():
    if not os.path.exists(args.room_path):
        print(f"Path {args.room_path} does not exist")
        return
    
    # file format: room_{i}.ply
    
    room_pcds = []
    for file in os.listdir(args.room_path):
        room_pcd = o3d.io.read_point_cloud(os.path.join(args.room_path, file))
        room_pcds.append(room_pcd)
    
    total_rooms = len(room_pcds)
        
    colors = generate_color_palette(total_rooms)
    
    for i in range(total_rooms):
        existing_colors = np.asarray(room_pcds[i].colors)
        new_colors = existing_colors * (1 - args.color_alpha) + np.array(colors[i]) * args.color_alpha
        room_pcds[i].colors = o3d.utility.Vector3dVector(new_colors)
        
    o3d.visualization.draw_geometries(room_pcds)

if __name__ == '__main__':
    args = parser.parse_args()
    
    main()