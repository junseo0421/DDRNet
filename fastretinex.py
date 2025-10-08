import numpy as np
import cv2
import pywt
import time

def msr(image, scales, weights):
    result = np.zeros_like(image, dtype=np.float32)
    for i, sigma in enumerate(scales):
        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        result += weights[i] * (np.log1p(image) - np.log1p(gaussian))
    return result

def fast_msr(image, scales, weights):
    v_msr = msr(image, scales, weights)
    v_msr = cv2.normalize(v_msr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return v_msr

def apply_wavelet_transform(channel):
    coeffs2 = pywt.dwt2(channel, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, (LH, HL, HH)

def inverse_wavelet_transform(LL, coeffs):
    return pywt.idwt2((LL, coeffs), 'haar')

def enhance_image(image_path, output_path):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply Haar DWT on the V channel
    v_LL, v_coeffs = apply_wavelet_transform(v)

    # Apply MSR to the LL part of the wavelet transform
    scales = [10, 60, 90, 160]
    weights = [0.25, 0.25, 0.25, 0.25]
    v_msr = fast_msr(v_LL, scales, weights)

    # Inverse wavelet transform
    v_enhanced = inverse_wavelet_transform(v_msr, v_coeffs)

    # Clip values to ensure they are in valid range
    v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)

    # Merge back the enhanced V channel with original H and S channels
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # Save the result
    cv2.imwrite(output_path, rgb_enhanced)


## enhance_image(이미지경로, 저장할이미지 경로 )
enhance_image('./dataset/aug_test/clear.jpg', './aug_out/enhanced.jpg')