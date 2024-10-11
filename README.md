
---

# Image Processing Lab with Gradio and OpenCV

This project provides an interactive web-based application for image processing, allowing users to apply various filters, transformations, and morphological operations to images. The app is built using [Gradio](https://gradio.app/) and OpenCV, with additional support from libraries like Scikit-Image and Matplotlib.

## Features

- **Filter Application**: Apply filters like grayscale, Gaussian blur, binarization, and negative to the uploaded images.
- **Contrast and Gamma Correction**: Adjust the contrast of the image and apply gamma transformation.
- **Morphological Operations**: Perform erosion and dilation on images to manipulate their structure.
- **Rotation and Resizing**: Rotate images by a specified degree and resize them based on a given scale.
- **Feature Extraction**: Detect contours and edges in the image using computer vision techniques.
- **Histogram Display**: Visualize the histogram of the image to better understand its pixel intensity distribution.
- **Image Reset**: Reset the image to its original state at any time.

## Installation

To run this project locally, you need Python 3.10+ and the following dependencies:

```bash
pip install gradio opencv-python scikit-image matplotlib
```

## Project folder Structure

- **`app.py`**: Contains the main Gradio interface and functions for applying image transformations.
- **`improcess.py`**: Contains utility functions for image processing (filters, transformations, morphological operations, etc.).
- **`profile.jpeg`**: Default image used for demonstration purposes.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Jugggernault/image_processing
   cd image_processing/PROJECT
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Access the Gradio interface at the local address provided in the terminal.

## Available Image Processing Functions

### Filters

- **Gray Scale**: Converts the image to grayscale.
- **Gaussian Blur**: Blurs the image using a Gaussian kernel.
- **Binarize**: Converts the image into a binary format based on pixel intensity.
- **Negative**: Creates the negative of the image.

### Transformations

- **Gamma Transformation**: Adjust the brightness using gamma correction.
- **Log Transformation**: Applies logarithmic transformation to compress pixel intensity.
- **Exponential Transformation**: Enhances brightness through exponential scaling.
- **Contrast Adjustments**: Apply dark or white contrast enhancements.

### Morphological Operations

- **Erosion**: Reduces the size of objects in the image.
- **Dilation**: Increases the size of objects in the image.

### Geometric Transformations

- **Rotation**: Rotate the image by a specified degree.
- **Resize**: Resize the image based on a specified scale factor.

### Feature Detection

- **Contours**: Detects and draws contours around objects in the image.
- **Edges**: Uses the Canny edge detector to find edges in the image.

### Histogram Equalization

- **Equalize**: Equalizes the image histogram to improve contrast.

### Histogram Visualization

- Display the image histogram to analyze the pixel intensity distribution.

## Example Usage

Upload an image, select a filter, adjust contrast, or apply a morphological operation, and the transformed image will be displayed. You can reset the image to its original state or visualize the histogram.

## License

This project is licensed under the MIT License.

---