import glob
import json
import torch
import numpy as np
import cv2 as cv
from PIL import Image
from lime import lime_image
from six import BytesIO
from skimage.segmentation import mark_boundaries
from ultralytics import YOLO

yolo_v8_cls_wbc_model = YOLO('src/resources/yolo_cls_wbc/weights.pt')
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
    # batch = np.stack(processed_images, axis=0)
    results = yolo_v8_cls_wbc_model.predict(processed_images)
    # Extract probabilities from each Results object
    # Extract probabilities from each Results object and convert them to numpy arrays
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


def lime_superpixels_plot(img, temp, mask, label, cell_type, capsule_code):
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap

    # Get the segmentation regions
    segments = explanation.segments

    # Get superpixel weights
    superpixel_weights = explanation.local_exp[label]

    # Sort superpixels by their index
    sorted_superpixel_weights = sorted(superpixel_weights, key=lambda x: x[0])
    superpixel_weights = sorted_superpixel_weights

    # Calculate the total absolute weight
    total_abs_weight = sum(abs(weight) for _, weight in superpixel_weights)
    # Normalize the weights
    norm_weights = {
        superpixel: weight / total_abs_weight for superpixel, weight in superpixel_weights
    }

    # Create a colormap for superpixel regions
    # cmap = plt.cm.viridis
    cmap = ListedColormap(plt.cm.tab10.colors[1:5])
    colored_segments = np.zeros_like(temp, dtype=np.float32)

    for superpixel, weight in superpixel_weights:
        mask = segments == superpixel
        colored_segments[mask] = cmap(superpixel)[:3]  # RGB from colormap

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
    plt.title(f"Positive vs Negative for {cell_types[i]}")
    plt.axis("off")

    # Plot the segmented image with weights
    plt.subplot(1, 3, 3)
    plt.imshow(colored_segments)
    plt.imshow(img, alpha=0.4)  # Overlay the original image
    plt.title(f"Superpixel for {cell_types[i]}")
    plt.axis("off")

    # Create a legend
    legend_elements = [
        Patch(facecolor=cmap(superpixel)[:3], edgecolor='black',
              label=f"{superpixel}: {norm_weights[superpixel]:.4f}")
        for superpixel, weight in superpixel_weights
    ]
    # Add the legend to the plot, ensuring one row per superpixel
    plt.legend(
        handles=legend_elements,
        loc='upper left',  # Position the legend
        bbox_to_anchor=(1.05, 1),  # Place legend outside the plot
        frameon=True,  # Add a border for clarity
        title="Superpixel Weights",  # Optional title for legend
        ncol=1,  # Force one column for the legend
        # alignment="left",  # Align text to the left
    )

    plt.tight_layout()
    # plt.savefig(f"test/xai/output/lime_label_{label}.png")
    plt.savefig(f"{output_folder}{cell_type}/{capsule_code}_class_{i}.png")
    plt.clf()

    print("image saved")


cell_types = ["monocytes", "lymphocytes", "eosinophils", "neutrophils"]

if __name__ == '__main__':

    output_folder = "/home/bruno/Documents/ai/aguia/experiments/human_bias/lime_xai/"
    images_folder = "/home/bruno/Documents/ai/aguia/experiments/human_bias/cropped_cells/12/"

    for cell_type in cell_types:

        base_folder = images_folder + cell_type + "/"
        img_files = glob.glob(base_folder + "*.png")

        for filename in img_files:
            cell_image = cv.imread(filename)

            # Save the final combined image as a single file
            capsule_code = filename.replace(base_folder, "").replace(".png", "")

            image_to_explain = cell_image
            explanation = explainer.explain_instance(
                image_to_explain,
                predict_fn,
                batch_size=100,
                top_labels=4,  # Set to the number of top labels you want to explain
                hide_color=0,
                num_samples=1000  # Number of perturbed images to generate
            )

            # 5. Visualize the explanation
            from matplotlib import pyplot as plt

            images = []
            for i in range(4):
                # explanation.top_labels[0]
                temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=100,
                                                            hide_rest=False)

                lime_superpixels_plot(image_to_explain, temp, mask, i, cell_type, capsule_code)

                # Example to retrieve and print superpixel importance for a particular label
                for label in explanation.top_labels:
                    print(f"Relevance for class {class_names[label]}:")
                    superpixel_weights = explanation.local_exp[label]
                    for superpixel, weight in superpixel_weights:
                        print(f"Superpixel {superpixel} has weight {weight}")

                # To get segmentation regions
                segments = explanation.segments
                print("Segmentation superpixels (shape):", segments.shape)

                boundaries2 = mark_boundaries(temp / 255, mask)
                plt.imshow(boundaries2)
                plt.axis('off')
                plt.title(f"{class_names[i]}")
                # plt.show()
                # plt.savefig(f"test/output/xai/lime_{i}_{class_names[i]}.png")
                # plt.clf()
                img_bytes = BytesIO()
                plt.savefig(img_bytes, format='PNG')
                images.append(Image.open(img_bytes))
                plt.clf()

            gray_img = preprocess_image(image_to_explain)
            plt.imshow(gray_img)
            plt.axis('off')
            plt.title(f"Base Gray")
            # plt.show()
            # plt.savefig(f"test/output/xai/lime_base_gray.png")
            # plt.clf()
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='PNG')
            images.append(Image.open(img_bytes))
            plt.clf()

            plt.imshow(image_to_explain)
            plt.axis('off')
            plt.title(f"Base RGB")
            # plt.show()
            # plt.savefig(f"test/output/xai/lime_base_rgb.png")
            # plt.clf()
            img_bytes = BytesIO()
            plt.savefig(img_bytes, format='PNG')
            images.append(Image.open(img_bytes))
            plt.clf()

            # Create a single large image combining all images in a grid (e.g., 2x3)
            grid_width = 3  # Adjust based on the desired number of columns
            grid_height = 2
            single_img_width, single_img_height = images[0].size  # Assuming all images have the same dimensions

            combined_image = Image.new(
                'RGB',
                (grid_width * single_img_width, grid_height * single_img_height)
            )

            # Paste each image into the correct position in the grid
            for index, img in enumerate(images):
                x = (index % grid_width) * single_img_width
                y = (index // grid_width) * single_img_height
                combined_image.paste(img, (x, y))

            combined_image.save(f"{output_folder}{cell_type}/{capsule_code}.png")
