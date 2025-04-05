

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import *
import colorsys
# import mixbox  # Import mixbox for paint mixing
import json
import math

paint_colors = [
    {"name": "phthalo-blue", "rgb": (0, 15, 137)},
    {"name": "titanium-white", "rgb": (255, 255, 255)},
    {"name": "cadmium-red", "rgb": (210, 36, 30)},
    {"name": "cadmium-yellow", "rgb": (255, 223, 81)},
    {"name": "burnt-umber", "rgb": (50, 5, 0)},
    {"name": "raw-umber", "rgb": (102, 76, 51)},
    {"name": "cerulean-blue", "rgb": (42, 82, 190)},
    {"name": "dioxazine-purple", "rgb": (91, 33, 115)},
    {"name": "quinacridone magenta", "rgb": (161, 0, 70)},
    {"name": "ivory black", "rgb": (0, 0, 0)},
    {"name": "sap green", "rgb":(36, 56, 15)}
]


def rgb_to_lab(rgb):
    rgb_arr = np.uint8([[rgb]])  # shape (1,1,3)
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    return lab[0][0]  # flatten to shape (3,)

def mix_lab_colors(lab_colors, weights):
    mixed = np.zeros(3)
    for color, weight in zip(lab_colors, weights):
        mixed += np.array(color) * weight
    return mixed

def lab_to_rgb(lab):
    lab_arr = np.uint8([[lab]])  # shape (1,1,3)
    rgb = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)
    return tuple(rgb[0][0])


def select_pixel_color(event, x, y, flags, param):
    """ Callback function to pick color from an image on click """
    if event == cv2.EVENT_LBUTTONDOWN:
        color = img[y, x]
        color_rgb = (int(color[2]), int(color[1]), int(color[0]))  
        # color_hsv = rgb_to_hsv(color_rgb)

        print(f"\n🎨 Selected Color: RGB {color_rgb}")
        # mix_suggestion = calculate_mix_percentages(color_rgb)

        # print("\n🖌 Suggested Paint Mix:")
        # for color_name, percent in mix_suggestion:
        #     print(f"{color_name} - {percent}%")

        # print("\n🔹 Adjust for brightness: Add Titanium White if needed")

        master = Tk()
        # Label(master, text='Color Palette')
        Label(master, text='Target Color').grid(row=0)

        e1 = Entry(master, bg=f"#{rgb_to_hex(color_rgb[0], color_rgb[1], color_rgb[2])}")

        e1.grid(row=0, column=1)
        final_mixes = []
        mix, names, percentages = get_paint_mix(color_rgb, paint_colors)
        final_mixes.append((mix, names, percentages))
        print('mix: ', final_mixes)
        
        return final_mixes

      

def rgb_to_hex(r, g, b):
  return ('{:02X}' * 3).format(r, g, b)

def extract_dominant_colors(image_path, num_colors=8):
    """ Extract the most dominant colors from an image """
    
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

    # Convert from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    image = np.float32(image)

    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10)
    kmeans.fit(image)
    rgb_colors = kmeans.cluster_centers_.astype(int)

    print('Extracted Dominant Colors: ', rgb_colors)

    return rgb_colors


def color_distance(rgb1, rgb2):
    if not isinstance(rgb2, tuple) or len(rgb2) != 3:
        raise ValueError(f"Invalid color format: {rgb2}. It should be a tuple of (R, G, B).")
    
    rgb1 = tuple(map(int, rgb1))
    rgb2 = tuple(map(int, rgb2))
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))



def calculate_ratios(color_distances):
    total_distance = sum(color_distances)
    return [1 - (distance / total_distance) for distance in color_distances]

def normalize_percentages(percentages):
    total = sum(percentages)
    return [round(p / total * 100, 2) for p in percentages]

def get_paint_mix(target_color, available_colors, max_paints=3):
    # Calculate distances for all available colors
    distances = [color_distance(target_color, color['rgb']) for color in available_colors]
    
    # Sort by closest to farthest
    closest_colors = sorted(zip(available_colors, distances), key=lambda x: x[1])
    
    # Select the closest 1-3 colors based on max_paints
    closest_colors = closest_colors[:max_paints]
    
    # Calculate the mix of the closest selected colors
    rgb_mix = [0, 0, 0]
    paint_names = []
    paint_rgb = []
    percentages = []
    
    # Total distance to calculate ratio
    total_distance = sum([dist for _, dist in closest_colors])
    
    for idx, (color, dist) in enumerate(closest_colors):

        if idx == 0:
            ratio = 1 - (dist / total_distance) * 0.01  # Further increase weight for the first color
        else:
            ratio = 1 - (dist / total_distance)  
            
        # Ensure the ratio is non-negative
        ratio = max(0, ratio)
        
        # Proportional mixing based on distance
        rgb_mix = [rgb_mix[i] + ratio * color['rgb'][i] for i in range(3)]
        paint_names.append(color['name'])
        paint_rgb.append(color['rgb'])
        percentages.append(ratio * 100)  
    

    percentages = normalize_percentages(percentages)
    
    # Normalize the mix to fit into the RGB range
    rgb_mix = [min(255, max(0, int(value))) for value in rgb_mix]
    
    return tuple(rgb_mix), paint_names, paint_rgb, percentages


# def test_paint_mix( rgbs, percentages):
#     rgb1 = (rgbs[0])
#     rgb2 = (rgbs[1])
#     rgb3 = (rgbs[2])

#     percentage1 = percentages[0]/100
#     percentage2 = percentages[0]/100
#     percentage3 = percentages[0]/100

#     z1 = mixbox.rgb_to_latent(rgb1)
#     z2 = mixbox.rgb_to_latent(rgb2)
#     z3 = mixbox.rgb_to_latent(rgb3)

#     z_mix = [0] * mixbox.LATENT_SIZE

#     for i in range(len(z_mix)):    
#         z_mix[i] = (percentage1*z1[i] +    
#                     percentage2*z2[i] +    
#                     percentage3*z3[i])      

#     rgb_mix = mixbox.latent_to_rgb(z_mix)

#     return rgb_mix
def test_paint_mix(rgbs, percentages):
    lab_colors = [rgb_to_lab(rgb) for rgb in rgbs]
    weights = [p / 100 for p in percentages]

    mixed_lab = mix_lab_colors(lab_colors, weights)
    mixed_rgb = lab_to_rgb(mixed_lab)
    
    return mixed_rgb


def get_paint_mixes(target_rgbs):
    final_mixes = []
    final_color = []
    for target_rgb in target_rgbs:
        print(f"🎨 Mixing for target rgb: {target_rgb}")
        mix, names, rgbs, percentages = get_paint_mix(target_rgb, paint_colors)
        final_mixes.append((mix, names, rgbs, percentages))
        singular_final_color= test_paint_mix(rgbs, percentages)
        final_color.append(singular_final_color)
        print ('paint names: ', names, ' percentages: ', percentages)
        print ('final color: ', singular_final_color)
    
    return final_mixes, final_color




def open_image_picker():
    """ Open a file picker to select an image """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path



# if __name__ == "__main__":
#     print('Hello! Welcome to the paint mixer')
    
#     # Open image selection dialog
#     image_path = open_image_picker()

#     if image_path:
#         print("\n📸 Extracting dominant colors from image...")

#         dominant_rgb = extract_dominant_colors(image_path)

#         print("\n🎨 Extracted Palette:")
#         for idx, color in enumerate(dominant_rgb):
#             print(f"Color {idx+1}: RGB {color}")

#         print("\n🎨 Suggested Paint Mixes:")
#         suggested_mix, final_color = get_paint_mixes(dominant_rgb, paint_colors)


#         # img = cv2.imread(image_path)
#         # img = cv2.resize(img, (600, 400))
#         # cv2.imshow("Click to Pick a Color", img)
#         # cv2.setMouseCallback("Click to Pick a Color", select_pixel_color)
#         # cv2.waitKey(0)

#         # 🖼 Tkinter UI
#         master = Tk()
#         master.title("Dominant Colors Palette")
#         Label(master, text='Color Palette').grid(row=0, columnspan=2)

#         for i, color in enumerate(dominant_rgb):
#             e = Entry(master, bg=f"#{rgb_to_hex(color[0], color[1], color[2])}")
#             e.grid(row=i+1, column=1)

#         for i, color in enumerate(final_color):
#             e = Entry(master, bg=f"#{rgb_to_hex(color[0], color[1], color[2])}")
#             e.grid(row=i+1, column=2)

#         master.mainloop()

