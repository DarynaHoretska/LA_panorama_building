# Panorama Construction via Linear Algebra

> A panorama stitching pipeline built entirely on linear algebra — homogeneous coordinates, DLT, SVD, and Hartley normalisation.

**Pechenenko Yaryna · Horetska Daryna · Vasylcheko Vladyslav**

---

## How it works

The core idea: the geometric relationship between two photos taken from the same spot (but different angles) is a **projective transformation** — a 3×3 matrix **H** called a homography. Finding it reduces to pure linear algebra:

1. **Homogeneous coordinates** — lift each pixel $(x, y)$ to $(x, y, 1)^\top$ so the projective map becomes a matrix–vector product $\tilde{x}' \sim H\tilde{x}$
2. **DLT** — each matched point pair gives 2 linear equations in the 9 entries of H, assembling $A\mathbf{h} = \mathbf{0}$, $A \in \mathbb{R}^{2n \times 9}$
3. **SVD** — the solution $\hat{\mathbf{h}}$ is the last column of $V$ in $A = U\Sigma V^\top$ (right singular vector for the smallest singular value)
4. **Hartley normalisation** — shifts each point set to zero centroid and scales to mean distance $\sqrt{2}$, applied as a change of basis $H = T'^{-1} H_n T$; reduces reprojection error from ~12 px to ~0.3 px
5. **RANSAC** — discards outlier matches before estimation
6. **Inverse warping + bilinear interpolation** — maps every output pixel back through $H^{-1}$
7. **Feathered blending** — distance-weighted average in the overlap zone



---

## Getting started

**Install dependencies:**
```bash
pip install numpy opencv-python
```

**Stitch two images:**
```bash
python panorama_stitching.py image1.jpg image2.jpg
```

**Stitch multiple images:**
```bash
python panorama_stitching.py img1.jpg img2.jpg img3.jpg --output panorama.jpg
```

**Optional flags:**
```bash
--output panorama.jpg   # output filename (default: panorama.jpg)
--method sift           # feature detector: sift or orb (default: sift)
--no-ransac             # skip RANSAC (not recommended)
--test                  # run synthetic validation test
```

---

## Results

### Example 1

| image1 | image2 |
|----------|----------|
| <img width="500" alt="image1" src="https://github.com/user-attachments/assets/5ec89bb3-f72f-45ee-9be1-63916027f4f6" /> | <img width="500" alt="image1" src="https://github.com/user-attachments/assets/6c5e2eb7-4e4c-4c3a-bd8f-a9b2f05ee89e" /> |

**Result:**

<img width="1000" alt="panorama result 1" src="https://github.com/user-attachments/assets/1831c172-fb5c-4470-8630-e4f8fc00c0b3" />

---

### Example 2

| image1 | image2 |
|----------|----------|
| <img width="500" alt="image" src="https://github.com/user-attachments/assets/df0bf638-6a5a-4103-8ed9-659a9c6fedfd" /> | <img width="500" alt="image" src="https://github.com/user-attachments/assets/38aa7709-f416-4384-9b3a-cf846d08688d" /> |

**Result:**

<img width="700" alt="panorama result 2" src="https://github.com/user-attachments/assets/62821900-6c55-43f2-b53f-350a4a9c57a3" />

---

### Example 3

| image1 | image2 |
|----------|----------|
| <img width="500" alt="image" src="https://github.com/user-attachments/assets/dba84615-abf2-43ee-b456-d582f32c446c" /> | <img width="500" alt="image" src="https://github.com/user-attachments/assets/12d49b82-987b-4298-9fc0-b5d37a2a35dc" /> |

**Result:**

<img width="700" alt="panorama result 3" src="https://github.com/user-attachments/assets/b0e569a4-5286-48eb-aaa9-1d51a9d8e8a4" />

---

### Multiple image stitching

| image1 | image2 | image3 |
|----------|----------|----------|
| <img width="500" alt="image" src="https://github.com/user-attachments/assets/55f8647e-ce99-47e5-8327-587f71f0d08c" /> | <img width="500" alt="image" src="https://github.com/user-attachments/assets/48f4ec29-067b-40b4-b062-bb6c1325d1b2" /> | <img width="500" alt="image" src="https://github.com/user-attachments/assets/a1c088ef-ba14-4280-ac5a-dc846f511a72" /> |

**Result:**

<img width="700" alt="panorama multi result" src="https://github.com/user-attachments/assets/28a45e21-9f1e-4b2f-8c48-12e249358964" />

---

## Video explanations

| Author | Video |
|---|---|
| Vasylchenko Vlad | [▶ Watch](https://youtu.be/Waf98fpNs1E) |
| Horetska Daryna | [▶ Watch](https://youtu.be/pKy9tjriv5M) |
| Pechenenko Yaryna | [▶ Watch](https://youtu.be/sQtxfVANVlY?si=7oAZFAnn383HZ6Lf) |

---

## References

1. Hartley — *In defense of the eight-point algorithm*
2. Fischler & Bolles — *Random sample consensus*
3. Lowe — *Distinctive image features from scale-invariant keypoints*
4. DeTone, Malisiewicz, Rabinovich — *SuperPoint: Self-supervised interest point detection and description*
5. Szeliski — *Computer Vision: Algorithms and Applications*
