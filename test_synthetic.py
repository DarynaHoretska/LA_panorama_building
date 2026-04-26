import numpy as np
import sys
sys.path.insert(0, ".")
from panorama_stitching import estimate_homography, ransac_homography

def apply_homography(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]

def reprojection_error(H, pts1, pts2):
    proj = apply_homography(H, pts1)
    errors = np.linalg.norm(proj - pts2, axis=1)
    return errors.mean(), errors.max()

def dlt_no_norm(pts1, pts2):
    n = len(pts1)
    A = np.zeros((2 * n, 9))
    for i, ((x, y), (xp, yp)) in enumerate(zip(pts1, pts2)):
        A[2*i]   = [0, 0, 0, -x, -y, -1,  yp*x,  yp*y,  yp]
        A[2*i+1] = [x, y, 1,  0,  0,  0, -xp*x, -xp*y, -xp]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

np.random.seed(42)
N, sigma = 50, 1.0

H_star = np.array([[1.2,  0.3,  400.0],
                   [0.1,  1.15, 200.0],
                   [0.0008, 0.0005, 1.0]])

pts1 = np.random.uniform(0, 3000, (N, 2)) 
pts2_clean = apply_homography(H_star, pts1)
pts2_noisy = pts2_clean + np.random.randn(N, 2) * sigma


H1 = dlt_no_norm(pts1, pts2_noisy)
m1, mx1 = reprojection_error(H1, pts1, pts2_noisy)


H2 = estimate_homography(pts1, pts2_noisy)
m2, mx2 = reprojection_error(H2, pts1, pts2_noisy)


n_out = int(0.2 * N)
pts2_ransac = pts2_noisy.copy()
pts2_ransac[:n_out] += np.random.randn(n_out, 2) * 50  
H3, inliers = ransac_homography(pts1, pts2_ransac)
m3, mx3 = reprojection_error(H3, pts1[inliers], pts2_ransac[inliers])

print("=" * 55)
print(f"{'Method':<35} {'Mean':>6} {'Max':>7}")
print("-" * 55)
print(f"{'DLT without normalisation':<35} {m1:>6.2f} {mx1:>7.2f}")
print(f"{'DLT with Hartley norm':<35} {m2:>6.2f} {mx2:>7.2f}")
print(f"{'DLT + Hartley + RANSAC':<35} {m3:>6.2f} {mx3:>7.2f}")
print("=" * 55)
print(f"RANSAC inliers: {inliers.sum()}/{N}")