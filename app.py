import cv2
import numpy as np
import gradio as gr

# Additional filter functions
def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

def apply_invert_filter(frame):
    return cv2.bitwise_not(frame)

def adjust_brightness_contrast(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_sepia_filter(frame):
    sepia_filter = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    return cv2.transform(frame, sepia_filter)

def apply_fall_filter(frame):
    fall_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    return cv2.transform(frame, fall_filter)

# New filter functions
def apply_vintage_filter(frame):
    # Convert to float for processing
    frame_float = frame.astype(float)
    
    # Adjust color balance for vintage look
    frame_float[:,:,0] *= 1.3  # Boost blue channel
    frame_float[:,:,2] *= 0.8  # Reduce red channel
    
    # Add slight sepia tone
    frame_float = cv2.transform(frame_float, np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ]))
    
    # Add vignette effect
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    # Apply vignette
    for i in range(3):
        frame_float[:,:,i] *= mask
    
    return np.clip(frame_float, 0, 255).astype(np.uint8)

def apply_pencil_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=5)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    return sketch

def apply_cartoon_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_watercolor(frame):
    # Bilateral filter with strong parameters for painting-like effect
    bilateral = cv2.bilateralFilter(frame, 9, 150, 150)
    # Enhance edges
    edge = cv2.edgePreservingFilter(bilateral, flags=1, sigma_s=60, sigma_r=0.6)
    return edge

# Main filter application function
def apply_filter(filter_type, input_image=None, intensity=1.0):
    if input_image is not None:
        frame = input_image
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Web kameradan görüntü alınamadı"

    # Apply selected filter
    filter_map = {
        "Gaussian Blur": lambda: apply_gaussian_blur(frame),
        "Sharpen": lambda: apply_sharpening_filter(frame),
        "Edge Detection": lambda: apply_edge_detection(frame),
        "Invert": lambda: apply_invert_filter(frame),
        "Brightness": lambda: adjust_brightness_contrast(frame, alpha=intensity, beta=50),
        "Grayscale": lambda: apply_grayscale_filter(frame),
        "Sepia": lambda: apply_sepia_filter(frame),
        "Sonbahar": lambda: apply_fall_filter(frame),
        "Vintage": lambda: apply_vintage_filter(frame),
        "Pencil Sketch": lambda: apply_pencil_sketch(frame),
        "Cartoon": lambda: apply_cartoon_effect(frame),
        "Watercolor": lambda: apply_watercolor(frame)
    }

    return filter_map.get(filter_type, lambda: frame)()

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📸 Advanced Photo Filter Application
    
    Upload an image or use your webcam to apply various artistic filters!
    
    ### Instructions:
    1. Select a filter from the dropdown menu
    2. Upload an image or capture from webcam
    3. Adjust intensity (for supported filters)
    4. Click 'Apply Filter' to see the result
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Filter selection and controls
            filter_type = gr.Dropdown(
                label="Select Filter",
                choices=[
                    "Gaussian Blur", "Sharpen", "Edge Detection", "Invert",
                    "Brightness", "Grayscale", "Sepia", "Sonbahar",
                    "Vintage", "Pencil Sketch", "Cartoon", "Watercolor"
                ],
                value="Gaussian Blur"
            )
            
            intensity = gr.Slider(
                label="Filter Intensity",
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
            
            apply_button = gr.Button("Apply Filter", variant="primary")

        with gr.Column(scale=2):
            # Image display area
            with gr.Row():
                input_image = gr.Image(label="Upload Image", type="numpy")
                output_image = gr.Image(label="Filtered Result")

    # Event handler
    apply_button.click(
        fn=apply_filter,
        inputs=[filter_type, input_image, intensity],
        outputs=output_image
    )

    gr.Markdown("""
    ### 🎨 Available Filters:
    - **Basic Filters**: Gaussian Blur, Sharpen, Edge Detection, Invert
    - **Color Filters**: Brightness, Grayscale, Sepia, Sonbahar
    - **Artistic Filters**: Vintage, Pencil Sketch, Cartoon, Watercolor
    """)

# Launch the interface
demo.launch()
