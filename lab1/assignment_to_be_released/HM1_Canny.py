import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float)
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """
    H, W = grad_mag.shape
    padded_grad_mag = np.zeros((H + 2, W + 2))
    padded_grad_mag[1:-1, 1:-1] = grad_mag

    grad_dir = np.mod(grad_dir, np.pi)
    dir_bins = np.pi / 8 * np.array([1.0, 3.0, 5.0, 7.0])
    mask_e_w = (grad_dir >= dir_bins[3]) | (grad_dir < dir_bins[0])
    mask_nw_se = (grad_dir >= dir_bins[0]) & (grad_dir < dir_bins[1])
    mask_n_s = (grad_dir >= dir_bins[1]) & (grad_dir < dir_bins[2])
    mask_ne_sw = (grad_dir >= dir_bins[2]) & (grad_dir < dir_bins[3])

    pixel_w = padded_grad_mag[1:-1, :-2]
    pixel_nw = padded_grad_mag[:-2, :-2]
    pixel_n = padded_grad_mag[:-2, 1:-1]
    pixel_ne = padded_grad_mag[:-2, 2:]
    pixel_e = padded_grad_mag[1:-1, 2:]
    pixel_se = padded_grad_mag[2:, 2:]
    pixel_s = padded_grad_mag[2:, 1:-1]
    pixel_sw = padded_grad_mag[2:, :-2]

    check_e_w = mask_e_w & (grad_mag > pixel_e) & (grad_mag > pixel_w)
    check_ne_sw = mask_ne_sw & (grad_mag > pixel_ne) & (grad_mag > pixel_sw)
    check_n_s = mask_n_s & (grad_mag > pixel_n) & (grad_mag > pixel_s)
    check_nw_se = mask_nw_se & (grad_mag > pixel_nw) & (grad_mag > pixel_se)

    check_all = check_e_w | check_ne_sw | check_n_s | check_nw_se

    NMS_output = np.zeros((H, W))
    NMS_output[check_all] = grad_mag[check_all]

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float)
        Outputs:
            output: array(float)
    """

    # you can adjust the parameters to fit your own implementation
    low_ratio = 0.3
    high_ratio = 0.6
    threshold = 0.3

    max_val = np.average(img[img > threshold]) * high_ratio
    min_val = np.average(img[img > threshold]) * low_ratio
    # max_val = img.max() * high_ratio
    # min_val = img.max() * low_ratio

    H, W = img.shape
    output = np.zeros((H, W))

    strong_edge = img >= max_val
    weak_edge = (img >= min_val) & (img < max_val)
    directions = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

    output[strong_edge] = 1.0
    stack = [(x, y) for x in range(H) for y in range(W) if strong_edge[x, y]]

    while stack:
        x, y = stack.pop()  # LIFO

        for dx, dy in directions:
            tx, ty = x + dx, y + dy
            if 0 <= tx < H and 0 <= ty < W and weak_edge[tx, ty]:
                output[tx, ty] = 1.0
                weak_edge[tx, ty] = False
                stack.append((tx, ty))

    return output


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("Lenna.png")/255

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    # Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(
        x_grad, y_grad)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    # write_img("result/HM1_Canny_middle.png", NMS_output*255)
    write_img("result/HM1_Canny_result.png", output_img*255)
