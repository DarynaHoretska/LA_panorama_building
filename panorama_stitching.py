import numpy as np
import cv2
import argparse
import sys
from pathlib import Path

def hartley_normalise(points: np.ndarray):
    centroid = points.mean(axis=0) 
    shifted  = points - centroid
    mean_distance = np.sqrt((shifted ** 2).sum(axis=1)).mean()

    scale = np.sqrt(2) / (mean_distance + 1e-10)

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    points_h    = np.column_stack([points, np.ones(len(points))])
    points_norm = (T @ points_h.T).T[:, :2]
    return points_norm, T

def build_A(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)

    for i, ((x, y), (x1, y1)) in enumerate(zip(src, dst)):
        A[2*i]     = [0,  0,  0, -x, -y, -1,  y1*x,  y1*y,  y1]
        A[2*i + 1] = [x,  y,  1,  0,  0,  0, -x1*x, -x1*y, -x1]

    return A

def estimate_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    assert len(src_points) >= 4, "Give more that 4 points"

    src_norm, T  = hartley_normalise(src_points)
    dst_norm, Tp = hartley_normalise(dst_points)

    A = build_A(src_norm, dst_norm)

    _, _, Vt = np.linalg.svd(A)
    h_hat = Vt[-1]
    H_norm = h_hat.reshape(3, 3)
    H = np.linalg.inv(Tp) @ H_norm @ T
    H /= H[2, 2]
    return H

def reprojection_error(H: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> float:
    n = len(src_points)
    src_h = np.column_stack([src_points, np.ones(n)])
    proj  = (H @ src_h.T).T
    proj_2d = proj[:, :2] / proj[:, 2:3]
    errors = np.linalg.norm(proj_2d - dst_points, axis=1)
    return float(errors.mean())


def ransac_homography(src_points: np.ndarray, dst_points: np.ndarray,
                      n_iter: int = 1000, threshold: float = 3.0):
    best_H = None
    best_inliers = np.zeros(len(src_points), dtype=bool)
    n_points = len(src_points)
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(n_points, 4, replace=False)
        try:
            H = estimate_homography(src_points[idx], dst_points[idx])
        except np.linalg.LinAlgError:
            continue

        src_h = np.column_stack([src_points, np.ones(n_points)])
        proj  = (H @ src_h.T).T
        w     = proj[:, 2:3]
        proj_2d = proj[:, :2] / np.where(np.abs(w) > 1e-8, w, 1e-8)
        err = np.linalg.norm(proj_2d - dst_points, axis=1)

        inliers = err < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_H       = H

    if best_inliers.sum() >= 4:
        best_H = estimate_homography(src_points[best_inliers], dst_points[best_inliers])

    return best_H, best_inliers

def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    if image.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]

    val = (image[y0c, x0c] * (1 - wx) * (1 - wy) +
           image[y0c, x1c] *      wx  * (1 - wy) +
           image[y1c, x0c] * (1 - wx) *      wy  +
           image[y1c, x1c] *      wx  *      wy)
    return val


def warp_image(image: np.ndarray, H: np.ndarray,
               canvas_shape: tuple) -> tuple:

    canvas_h, canvas_w = canvas_shape
    H_inv = np.linalg.inv(H)

    u, v = np.meshgrid(np.arange(canvas_w), np.arange(canvas_h))
    canvas_points = np.stack([u.ravel(), v.ravel(), np.ones(canvas_h * canvas_w)], axis=0)  # (3, N)

    src_points = H_inv @ canvas_points
    w_coord = src_points[2]
    valid   = np.abs(w_coord) > 1e-8
    x_src   = np.where(valid, src_points[0] / w_coord, -1)
    y_src   = np.where(valid, src_points[1] / w_coord, -1)

    img_h, img_w = image.shape[:2]
    inside = (valid & (x_src >= 0) & (x_src < img_w - 1) &
                      (y_src >= 0) & (y_src < img_h - 1))

    warped = np.zeros((canvas_h * canvas_w,) + image.shape[2:], dtype=np.float64)
    if inside.any():
        warped[inside] = bilinear_interpolate(
            image.astype(np.float64), x_src[inside], y_src[inside]
        )

    warped = warped.reshape(canvas_h, canvas_w, *image.shape[2:])
    mask   = inside.reshape(canvas_h, canvas_w)
    return warped.astype(np.uint8), mask

def distance_transform_weight(mask: np.ndarray) -> np.ndarray:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_d = dist.max()
    if max_d > 0:
        dist /= max_d
    return dist


def blend_images(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    w1 = distance_transform_weight(mask1)[:, :, None].astype(np.float64)
    w2 = distance_transform_weight(mask2)[:, :, None].astype(np.float64)

    overlap = (mask1 & mask2)
    only1   = (mask1 & ~mask2)
    only2   = (~mask1 & mask2)

    result = np.zeros_like(img1, dtype=np.float64)

    denom = w1 + w2 + 1e-10
    result[overlap] = (
        (w1 * img1.astype(np.float64) + w2 * img2.astype(np.float64)) / denom
    )[overlap]

    result[only1] = img1[only1].astype(np.float64)
    result[only2] = img2[only2].astype(np.float64)

    return np.clip(result, 0, 255).astype(np.uint8)


def detect_and_match(img1_gray: np.ndarray, img2_gray: np.ndarray,
                     method: str = "sift"):
    if method == "sift":
        try:
            detector = cv2.SIFT_create()
        except AttributeError:
            detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(img1_gray, None)
    kp2, des2 = detector.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ValueError("You need more points for stitching")

    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        raise ValueError(f"Not enough good matches: {len(good)} (must be ≥ 4)")

    src_points = np.float64([kp1[m.queryIdx].pt for m in good])
    dst_points = np.float64([kp2[m.trainIdx].pt for m in good])
    print(f"  {len(good)} matches found after the ratio test")
    return src_points, dst_points

def compute_canvas(img1: np.ndarray, img2: np.ndarray, H: np.ndarray):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.array([
        [0,  0,  1],
        [w2, 0,  1],
        [w2, h2, 1],
        [0,  h2, 1],
    ], dtype=np.float64).T 

    proj = (H @ corners2) 
    proj_2d = proj[:2] / proj[2] 

    corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float64).T

    all_points = np.hstack([corners1, proj_2d])      # (2, 8)
    x_min, y_min = np.floor(all_points.min(axis=1)).astype(int)
    x_max, y_max = np.ceil (all_points.max(axis=1)).astype(int)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    return (canvas_h, canvas_w), (-x_min, -y_min)

def stitch_pair(path1: str, path2: str, output_path: str = "panorama.jpg", method: str = "sift", use_ransac: bool = True) -> np.ndarray:
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None:
        raise FileNotFoundError(f"Can't open {path1}")
    if img2 is None:
        raise FileNotFoundError(f"Can't open {path2}")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    src_points, dst_points = detect_and_match(gray2, gray1, method=method)

    if use_ransac:
        H, inliers = ransac_homography(src_points, dst_points)
        print(f"  RANSAC: {inliers.sum()}/{len(inliers)} інлаєрів")
        err = reprojection_error(H, src_points[inliers], dst_points[inliers])
    else:
        H = estimate_homography(src_points, dst_points)
        err = reprojection_error(H, src_points, dst_points)
    print(f"  Reprojection error: {err:.4f} px")

    canvas_shape, (ox, oy) = compute_canvas(img1, img2, H)
    print(f"  Canvas: {canvas_shape[1]} × {canvas_shape[0]} px,  offset=({ox},{oy})")

    T_offset = np.array([
        [1, 0, ox],
        [0, 1, oy],
        [0, 0,  1],
    ], dtype=np.float64)

    H_canvas = T_offset @ H
    warped2, mask2 = warp_image(img2, H_canvas, canvas_shape)

    canvas1 = np.zeros((*canvas_shape, 3), dtype=np.uint8)
    canvas1[oy:oy + img1.shape[0], ox:ox + img1.shape[1]] = img1
    mask1 = np.zeros(canvas_shape, dtype=bool)
    mask1[oy:oy + img1.shape[0], ox:ox + img1.shape[1]] = True

    panorama = blend_images(canvas1, mask1, warped2, mask2)

    cv2.imwrite(output_path, panorama)
    return panorama


def stitch_multiple(image_paths: list, output_path: str = "panorama_full.jpg",
                    method: str = "sift", use_ransac: bool = True) -> np.ndarray:
    """
    Stitches a list of images sequentially from left to right.
    """
    if len(image_paths) < 2:
        raise ValueError("At least 2 images are required")

    temp_path = "_tmp_pano.jpg"

    print(f"\n--- Stitching image 1/ {len(image_paths)} with image 2/{len(image_paths)} ---")
    result = stitch_pair(
        image_paths[0],
        image_paths[1],
        output_path=temp_path,
        method=method,
        use_ransac=use_ransac,
    )

    for i in range(2, len(image_paths)):
        print(f"\n--- Adding image {i + 1}/{len(image_paths)} ---")
        result = stitch_pair(
            temp_path,
            image_paths[i],
            output_path=temp_path,
            method=method,
            use_ransac=use_ransac,
        )

    cv2.imwrite(output_path, result)
    print(f"\nFinal panorama saved to: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Panorama stitching via Linear Algebra (DLT + SVD)"
    )
    parser.add_argument("images", nargs="+",
                        help="Paths to images (from left to right)")
    parser.add_argument("--output", default="panorama.jpg",
                        help="Output file (default: panorama.jpg)")
    parser.add_argument("--method", choices=["sift", "orb"], default="sift",
                        help="Keypoint detector (default: sift)")
    parser.add_argument("--no-ransac", action="store_true",
                        help="Do not use RANSAC")
    parser.add_argument("--test", action="store_true",
                        help="Run the synthetic test and exit")
    args = parser.parse_args()

    if len(args.images) == 1:
        print("There should be at least 2 images")
        sys.exit(1)
    elif len(args.images) == 2:
        stitch_pair(
            args.images[0],
            args.images[1],
            output_path=args.output,
            method=args.method,
            use_ransac=not args.no_ransac,
        )
    else:
        stitch_multiple(
            args.images,
            output_path=args.output,
            method=args.method,
            use_ransac=not args.no_ransac,
        )


if __name__ == "__main__":
    main()