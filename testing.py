import numpy as np
from panorama_stitching import estimate_homography

def dlt_no_norm(pts1, pts2):
    n = len(pts1)
    A = np.zeros((2 * n, 9))
    for i, ((x, y), (xp, yp)) in enumerate(zip(pts1, pts2)):
        A[2*i]   = [0, 0, 0, -x, -y, -1,  yp*x,  yp*y,  yp]
        A[2*i+1] = [x, y, 1,  0,  0,  0, -xp*x, -xp*y, -xp]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def apply_H(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]

def err(H, p1, p2):
    return np.linalg.norm(apply_H(H, p1) - p2, axis=1)

np.random.seed(42)
N = 50

H_star = np.array([[1.2,  0.3,  400.0],
                   [0.1,  1.15, 200.0],
                   [0.0008, 0.0005, 1.0]])

pts1 = np.random.uniform(0, 3000, (N, 2))
pts2_clean = apply_H(H_star, pts1)

print(f"{'Noise σ':>10} | {'No norm mean':>14} {'max':>8} | {'Hartley mean':>14} {'max':>8} | {'Speedup':>8}")
print("-" * 75)

for sigma in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
    noise = np.random.randn(N, 2) * sigma
    pts2 = pts2_clean + noise

    H_no  = dlt_no_norm(pts1, pts2)
    H_yes = estimate_homography(pts1, pts2)

    e_no  = err(H_no,  pts1, pts2)
    e_yes = err(H_yes, pts1, pts2)

    ratio = e_no.mean() / (e_yes.mean() + 1e-12)
    print(f"{sigma:>10.1f} | {e_no.mean():>14.4f} {e_no.max():>8.4f} | "
          f"{e_yes.mean():>14.4f} {e_yes.max():>8.4f} | {ratio:>8.2f}x")