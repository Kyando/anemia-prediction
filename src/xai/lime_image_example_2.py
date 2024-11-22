import glob
import json
import os

import cv2 as cv
import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import slic
from ultralytics import YOLO

import src.cell_detector as cd
from src.detector import get_image_from_string
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

yolo_v8_cls_wbc_model = YOLO('src/resources/yolo_cls_wbc/weights.pt')


# Define a custom segmentation function
def custom_segmentation_fn(image):
    return slic(image, n_segments=30, compactness=10, sigma=0)


# Initialize LIME Image Explainer with the custom segmentation function
explainer = lime_image.LimeImageExplainer()

class_names = {0: 'eosinophils', 1: 'lymphocytes', 2: 'monocytes', 3: 'neutrophils'}


def preprocess_image(cell_img):
    img_yuv = cv.cvtColor(cell_img, cv.COLOR_BGR2YUV)
    y, u, v = cv.split(img_yuv)
    y = cv.cvtColor(y, cv.COLOR_GRAY2BGR)

    return y


# 3. Define a function to get predictions from YOLOv8 in the right format for LIME
def predict_fn(images):
    processed_images = [preprocess_image(img) for img in images]
    # YOLOv8 model expects a batch, convert images to a batch format
    results = yolo_v8_cls_wbc_model.predict(processed_images)

    # Extract probabilities from each Results object and convert them to numpy arrays
    probabilities = []
    for result in results:
        # Assuming `result.probs` is the attribute containing probability values
        if hasattr(result, 'probs'):
            probs = result.probs.data
            if isinstance(probs, torch.Tensor):  # Check if it's a tensor
                probs = probs.cpu().numpy()  # Move to CPU and convert to numpy array
            probabilities.append(probs)
        else:
            raise AttributeError("Expected 'probs' attribute in result object")

    return np.array(probabilities)  # Shape should be (num_images, num_classes)


def lime_superpixels_plot(img, temp, mask, label, cell_type, capsule_code, cell_id):

    # Get the segmentation regions
    segments = explanation.segments

    # Get superpixel weights
    superpixel_weights = explanation.local_exp[label]

    # Sort superpixels by their index
    sorted_superpixel_weights = sorted(superpixel_weights, reverse=True, key=lambda x: x[1])
    superpixel_weights = sorted_superpixel_weights

    # Calculate the total absolute weight
    total_abs_weight = sum(abs(weight) for _, weight in superpixel_weights)

    # Normalize the weights
    norm_weights = {
        superpixel: weight / total_abs_weight for superpixel, weight in superpixel_weights
    }

    # Create custom colormap for red-to-blue gradient
    cmap = LinearSegmentedColormap.from_list(
        "red_blue_cmap",
        [(.7, .2, .2), (1, 1, 1), (.2, .7, .2)],  # Red, White, Green
        N=256
    )

    # Map weights to colors
    colored_segments = np.zeros((*segments.shape, 3), dtype=np.float32)
    for superpixel, weight in superpixel_weights:
        mask = segments == superpixel
        # Normalize weight to [-1, 1] for coloring
        normalized_weight = max(-1, min(1, weight / total_abs_weight))
        normalized_weight = normalized_weight * 10
        color = cmap((normalized_weight + 1) / 2)[:3]  # Scale to [0, 1] for colormap
        colored_segments[mask] = color

    # Plot the image with superpixel boundaries
    plt.figure(figsize=(15, 10))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original - {cell_type}")
    plt.axis("off")

    # Plot the original image
    plt.subplot(1, 3, 2)
    plt.imshow(temp)
    plt.title(f"Positive vs Negative for {cell_types[label]}")
    plt.axis("off")

    # Plot the segmented image with weights
    plt.subplot(1, 3, 3)
    plt.imshow(colored_segments)
    plt.imshow(img, alpha=0.4)  # Overlay the original image
    plt.title(f"Superpixel for {cell_types[label]}")
    plt.axis("off")

    # Overlay superpixel numbers
    for superpixel, weight in superpixel_weights:
        # Find the coordinates of the superpixel
        y_coords, x_coords = np.where(segments == superpixel)
        if len(y_coords) > 0 and len(x_coords) > 0:
            # Calculate the center of the superpixel region
            y_center = int(np.mean(y_coords))
            x_center = int(np.mean(x_coords))
            # Add the superpixel number as text
            plt.text(
                x_center, y_center,
                str(superpixel),
                color="white", fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
            )

    # Create a legend
    legend_elements = [
        Patch(facecolor=cmap((((weight / total_abs_weight) * 10) + 1) / 2)[:3], edgecolor='black',
              label=f"{superpixel}: {norm_weights[superpixel] * 100:.2f}")
        for superpixel, weight in superpixel_weights
    ]
    # Add the legend to the plot
    plt.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        frameon=True,
        title="Superpixel Weights",
        ncol=1
    )

    plt.tight_layout()
    plt.savefig(f"{output_folder}{capsule_code}/{cell_type}/{cell_id}_{cell_types[label]}.png")
    plt.clf()

    print("Image saved")


cell_types = ["monocytes", "lymphocytes", "eosinophils", "neutrophils"]

if __name__ == '__main__':

    input_folder = "/home/bruno/Documents/ai/aguia/experiments/human_bias/base_payload/"
    output_folder = "/home/bruno/Documents/ai/aguia/experiments/human_bias/lime_xai/lime_capsule_output/"

    input_files = glob.glob(input_folder + "*.json")
    for input_file in input_files:
        input_json = json.loads(open(input_file).read())
        capsule_code = input_json['code']
        img = get_image_from_string(input_json['processed_white_sample'])
        final_result = input_json['finalResult']
        print(final_result)

        analyte_cells = {}
        for result in final_result:
            analyte_name = result.get("c4i0_name", "")
            if analyte_name in cell_types:
                cell_pos = result['cell_positions']
                cells_cropped, _, _ = cd.get_cells_cropped(img.copy(), cell_pos)
                analyte_cells[analyte_name] = cells_cropped

        print(analyte_cells.keys())

        for cell_type in cell_types:
            cropped_images = analyte_cells[cell_type]
            os.makedirs(f"{output_folder}{capsule_code}/{cell_type}", exist_ok=True)

            for cell_id, cell_image in enumerate(cropped_images):
                image_to_explain = cell_image
                explanation = explainer.explain_instance(
                    image_to_explain,
                    predict_fn,
                    batch_size=800,
                    top_labels=4,  # Set to the number of top labels you want to explain
                    hide_color=0,
                    num_samples=4000,  # Number of perturbed images to generate
                    segmentation_fn=custom_segmentation_fn,
                )

                # 5. Visualize the explanation
                images = []
                for i, _ in enumerate(cell_types):
                    # explanation.top_labels[0]
                    temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=100,
                                                                hide_rest=False)

                    lime_superpixels_plot(image_to_explain, temp, mask, i, cell_type, capsule_code, cell_id)
