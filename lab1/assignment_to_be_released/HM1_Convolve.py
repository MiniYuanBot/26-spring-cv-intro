import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    H, W = img.shape
    H_padded = H + 2 * padding_size
    W_padded = W + 2 * padding_size

    padding_img = np.zeros((H_padded, W_padded))

    if type == "zeroPadding":
        padding_img[padding_size:H_padded - padding_size,
                    padding_size:W_padded - padding_size] = img
    elif type == "replicatePadding":
        # center
        padding_img[padding_size:H_padded - padding_size,
                    padding_size:W_padded - padding_size] = img
        # edge
        padding_img[0:padding_size, padding_size:W + padding_size] = img[0:1, :]  # up
        padding_img[H + padding_size:, padding_size:W +
                    padding_size] = img[-1:, :]  # down
        padding_img[padding_size:H + padding_size,
                    0:padding_size] = img[:, 0:1]  # left
        padding_img[padding_size:H + padding_size,
                    W + padding_size:] = img[:, -1:]  # right
        # corner
        padding_img[0:padding_size, 0:padding_size] = img[0, 0]  # left up
        padding_img[0:padding_size, W + padding_size:] = img[0, -1]  # right up
        padding_img[H + padding_size:, 0:padding_size] = img[-1, 0]  # left down
        padding_img[H + padding_size:, W + padding_size:] = img[-1, -1]  # right down

    return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding
    K = kernel.shape[0]
    padding_size = (K - 1) // 2
    padding_img = padding(img, padding_size, "zeroPadding")  # (H_padded, W_padded)
    x = padding_img.ravel()  # (H_padded * W_padded,)

    H, W = img.shape
    H_padded = H + 2 * padding_size
    W_padded = W + 2 * padding_size

    # build the Toeplitz matrix T and compute convolution
    # Row i of matrix T specifies which positions in the flattened padded input
    # image are nedded to compute the i-th pixel of the flattened output image.
    T = np.zeros((H * W, H_padded * W_padded))

    # 第 (i, j) 元代表了窗口 (i, j) 元对应的输入图像像素与窗口 (0, 0) 元对应的
    # 输入图像像素在展平后的索引之差
    offsets = np.arange(K)[:, None] * W_padded + np.arange(K)[None, :]  # (K, K)
    offsets = offsets.ravel()

    # 第 (i, j) 元代表了输出图像像素 (i, j) 对应窗口的左上角位置在展平后的索引
    windows = np.arange(H)[:, None] * W_padded + np.arange(W)[None, :]  # (H, W)
    windows = windows.ravel()

    # 第 i * W + j 行代表了输出图像像素 (i, j) 计算所需的输入像素索引
    indices = windows[:, None] + offsets[None, :]  # (H * W, K * K)

    kernel_vec = kernel.ravel()  # (K * K,)

    T[np.arange(36)[:, None], indices] = kernel_vec

    y = T @ x
    output = y.reshape(H, W)

    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """

    # build the sliding-window convolution here
    H, W = img.shape
    K = kernel.shape[0]

    H_out = H - K + 1
    W_out = W - K + 1

    img_vec = img.ravel()
    kernel_vec = kernel.ravel()

    # 第 (i, j) 元代表了窗口 (i, j) 元对应的输入图像像素与窗口 (0, 0) 元对应的
    # 输入图像像素在展平后的索引之差
    offsets = np.arange(K)[:, None] * W + np.arange(K)[None, :]  # (K, K)
    offsets = offsets.ravel()

    # 第 (i, j) 元代表了输出图像像素 (i, j) 对应窗口的左上角位置在展平后的索引
    windows = np.arange(H_out)[:, None] * W + \
        np.arange(W_out)[None, :]  # (H_out, W_out)
    windows = windows.ravel()

    # 第 i * W_out + j 行代表了输出图像像素 (i, j) 计算所需的输入像素索引
    indices = windows[:, None] + offsets[None, :]  # (H_out * W_out, K * K)

    T = img_vec[indices]  # (H_out * W_out, K * K)

    output = T @ kernel_vec
    output = output.reshape(H_out, W_out)

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "zeroPadding")
    gaussian_kernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("Lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)
