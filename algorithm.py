import numpy as np
from numpy import ndarray


def plane(img: ndarray) -> ndarray:
    w = img.shape[0]
    h = img.shape[1]
    ret = ndarray((w * 3, h * 3), dtype=float)
    # final results all use float to express
    for i in range(0, 8):
        for x in range(w):
            for y in range(h):
                ret[x + i // 3 * w, y + (i % 3) * h] = 0 if ((img[x, y] & (1 << (7 - i))) == 0) else 1
    for x in range(w):
        for y in range(h):
            ret[x + 2 * w, y + 2 * h] = 1
    return ret


def equalize(img: ndarray) -> ndarray:
    if img.ndim == 2:
        return m_gray_equa(img)
    else:
        w = img.shape[0]
        h = img.shape[1]
        ret = ndarray((w, h, 3), dtype=float)
        for i in range(3):
            tmp = ndarray((w, h), dtype=float)
            for x in range(w):
                for y in range(h):
                    tmp[x, y] = img[x, y, i]
            tmp_res = m_gray_equa(tmp)
            for x in range(w):
                for y in range(h):
                    ret[x, y, i] = tmp_res[x, y]
        return ret


def m_gray_equa(img: ndarray) -> ndarray:
    w = img.shape[0]
    h = img.shape[1]
    count = ndarray((256,))
    m_map = ndarray((256,), dtype=float)
    for i in range(256):
        count[i] = 0
        m_map[i] = 0
    ret = ndarray(img.shape, dtype=float)
    for x in range(w):
        for y in range(h):
            count[int(img[x, y] * 255)] = count[int(img[x, y] * 255)] + 1
    total = sum(count)
    for i in range(256):
        for k in range(i + 1):
            m_map[i] = (m_map[i] + count[k])
        m_map[i] = m_map[i] / total
    for x in range(w):
        for y in range(h):
            ret[x, y] = m_map[int(img[x, y] * 255)]
    return ret


def denoise(img: ndarray) -> ndarray:
    ret = ndarray(img.shape, dtype=float)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            ret[x, y] = img[x, y]

    for x in range(0, img.shape[0] - 2):
        for y in range(0, img.shape[1] - 2):
            if img.ndim == 2:
                color_matrix = []
                for i in range(3):
                    for j in range(3):
                        color_matrix.append(img[x + i, y + j])
                ret[x + 1, y + 1] = np.sort(color_matrix)[5]
            else:
                for k in range(3):
                    color_matrix = []
                    for i in range(3):
                        for j in range(3):
                            color_matrix.append(img[x + i, y + j, k])
                    ret[x + 1, y + 1, k] = np.sort(color_matrix)[5]
    return ret


def interpolate(img: ndarray) -> ndarray:
    w = img.shape[0]
    h = img.shape[1]
    ret = ndarray((w * 2, h * 2), dtype=float) if img.ndim == 2 else ndarray((w * 2, h * 2, 3), dtype=float)
    for x in range(w * 2):
        for y in range(h * 2):
            if img.ndim == 2:
                ret[x, y] = bilinear_interpolation(img, x, y, 0)
            else:
                for i in range(3):
                    ret[x, y, i] = bilinear_interpolation(img, x, y, i)
    return ret


def bilinear_interpolation(img: ndarray, dst_x, dst_y, i) -> float:
    dst_x = dst_x + (1 if dst_x == 0 else 0) - (1 if dst_x == img.shape[0] * 2 - 1 else 0)
    dst_y = dst_y + (1 if dst_y == 0 else 0) - (1 if dst_y == img.shape[1] * 2 - 1 else 0)
    src_x = int((dst_x + 0.5) / 2 - 0.5)
    src_y = int((dst_y + 0.5) / 2 - 0.5)
    x = dst_x / 2 - src_x
    y = dst_y / 2 - src_y
    f00 = img[src_x, src_y] if img.ndim == 2 else img[src_x, src_y, i]
    f10 = img[src_x + 1, src_y] if img.ndim == 2 else img[src_x + 1, src_y, i]
    f01 = img[src_x, src_y + 1] if img.ndim == 2 else img[src_x, src_y + 1, i]
    f11 = img[src_x + 1, src_y + 1] if img.ndim == 2 else img[src_x + 1, src_y + 1, i]
    res = (f10 - f00) * x + (f01 - f00) * y + (f11 + f00 - f10 - f01) * x * y + f00
    res = 1 if res > 1 else res
    res = 0 if res < 0 else res
    return res


def dft(img: ndarray) -> ndarray:
    ret = np.log(np.abs(np.fft.fft2(centralize_gray(img))))
    return ret


def butterworth(img: ndarray) -> ndarray:
    w = img.shape[0]
    h = img.shape[1]
    f = np.fft.fft2(centralize_gray(img))
    d0 = 50
    guv = ndarray(img.shape, dtype='complex_')
    for i in range(w):
        for j in range(h):
            duv = np.sqrt((i - w / 2) ** 2 + (j - h / 2) ** 2)
            huv = 1 / (1 + (duv / d0) ** 4)
            guv[i, j] = huv * f[i, j]
    ret = centralize_gray(np.real(np.fft.ifft2(guv)))
    return ret


def centralize_gray(img: ndarray) -> ndarray:
    a = ndarray(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a[i, j] = img[i, j] * ((-1) ** (i + j))
    return a


def canny(img: ndarray) -> ndarray:
    w = img.shape[0]
    h = img.shape[1]

    gauss = np.zeros(img.shape, dtype='float')
    gauss_kernel = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    for x in range(0, w):
        for y in range(0, h):
            if 2 <= x < w-2 and 2 <= y < h-2:
                gauss[x, y] = 0
                for i in range(5):
                    for j in range(5):
                        gauss[x, y] += gauss_kernel[i, j]*img[x-2+i, y-2+j]
                gauss[x, y] /= 159
            else:
                gauss[x, y] = img[x, y]

    sobel_g = np.zeros(img.shape, dtype='float')
    sobel_theta = np.zeros(img.shape, dtype='float')
    for x in range(1, w-1):
        for y in range(1, h-1):
            gx = - (gauss[x-1, y-1] + 2 * gauss[x-1, y] + gauss[x-1, y+1]) + (gauss[x+1, y-1] + 2*gauss[x+1, y] + gauss[x+1, y+1])
            gy = - (gauss[x-1, y+1] + 2 * gauss[x, y+1] + gauss[x+1, y+1]) + (gauss[x-1, y-1] + 2*gauss[x, y-1] + gauss[x+1, y-1])
            sobel_g[x, y] = np.sqrt(gx*gx + gy*gy)
            if gx == 0:
                sobel_theta[x, y] = np.pi/2
            else:
                sobel_theta[x, y] = np.arctan(gy/gx)

    histogram = []
    inhibition = np.zeros(img.shape, dtype='float')
    for x in range(1, w-1):
        for y in range(1, h-1):
            if -np.pi < sobel_theta[x, y] * 8 <= np.pi:
                g_max = max(sobel_g[x+1, y], sobel_g[x-1, y])
            elif np.pi < sobel_theta[x, y] * 8 <= np.pi * 3:
                g_max = max(sobel_g[x+1, y-1], sobel_g[x-1, y+1])
            elif -np.pi * 3 < sobel_theta[x, y] * 8 <= -np.pi:
                g_max = max(sobel_g[x-1, y-1], sobel_g[x+1, y+1])
            else:
                g_max = max(sobel_g[x, y-1], sobel_g[x, y+1])
            inhibition[x, y] = 0 if sobel_g[x, y] <= g_max else sobel_g[x, y]
            if inhibition[x, y] > 0:
                histogram.append(inhibition[x, y])

    histogram.sort()
    ht = histogram[int(len(histogram)*0.65)]
    lt = 0.4 * ht

    res = np.zeros(img.shape, dtype='float')
    for x in range(1, w-1):
        for y in range(1, h-1):
            if inhibition[x, y] >= ht:
                res[x, y] = 1
            elif inhibition[x, y] < lt:
                res[x, y] = 0
            else:
                i_max = 0
                for i in range(3):
                    for j in range(3):
                        i_max = max(i_max, inhibition[x-1+i, y-1+j])
                if i_max > ht:
                    res[x, y] = 1
                else:
                    res[x, y] = 0

    return res


def morphology(img: ndarray) -> ndarray:
    ret = ndarray(img.shape, dtype='float')
    w = img.shape[0]
    h = img.shape[1]
    for t in range(w):
        for m in range(h):
            tmp = img[t, m]
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if 0 <= t + x < w and 0 <= m + y < h and -1 <= x+y <= 1:
                        tmp = min(tmp, img[t + x, m + y])
            ret[t, m] = tmp
    ret2 = ndarray(img.shape, dtype='float')
    for t in range(w):
        for m in range(h):
            ret2[t, m] = ret[t, m]
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if 0 <= t + x < w and 0 <= m + y < h and -1 <= x+y <= 1:
                        ret2[t, m] = max(ret2[t, m], ret[t + x, m + y])

    return ret2
