import os, json, glob
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
DATASET_DIR = "dataset"
CLASSES = ["pasto", "madera", "baldosa", "alfombra"] 
RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = 'uniform'
HSV_BINS = (8, 8, 8)   
DSCALE = 0.5           
OUT_MODEL_JSON = "centroides_superficies.json"

def features_color_hsv(bgr):
    if DSCALE != 1.0:
        bgr = cv2.resize(bgr, (0,0), fx=DSCALE, fy=DSCALE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None, HSV_BINS, [0,180, 0,256, 0,256]).flatten().astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0: hist /= hist_sum
    return hist  
def features_lbp(bgr, radius=RADIUS, n_points=N_POINTS):
    if DSCALE != 1.0:
        bgr = cv2.resize(bgr, (0,0), fx=DSCALE, fy=DSCALE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method=METHOD)
    bins = n_points + 2 if METHOD == 'uniform' else int(lbp.max()+1)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0: hist /= s
    return hist
def extract_feature(bgr):
    h = features_color_hsv(bgr)
    t = features_lbp(bgr)
    # concatenar y L2-normalizar para distancia euclidiana estable
    feat = np.concatenate([h, t]).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0: feat /= norm
    return feat
def main():
    class_feats = {c: [] for c in CLASSES}
    for c in CLASSES:
        pattern = os.path.join(DATASET_DIR, c, "*")
        for fp in glob.glob(pattern):
            img = cv2.imread(fp)
            if img is None: 
                continue
            feat = extract_feature(img)
            class_feats[c].append(feat.tolist())
    model = {"classes": CLASSES, "centroids": {}, "feat_dim": None,
             "params": {"HSV_BINS": HSV_BINS, "RADIUS": RADIUS, "N_POINTS": N_POINTS,
                        "METHOD": METHOD, "DSCALE": DSCALE}}
    for c, feats in class_feats.items():
        if len(feats) == 0:
            continue
        M = np.array(feats, dtype=np.float32)
        centroid = M.mean(axis=0)
        n = np.linalg.norm(centroid)
        if n > 0: centroid = centroid / n
        model["centroids"][c] = centroid.tolist()
        model["feat_dim"] = len(centroid)
    with open(OUT_MODEL_JSON, "w") as f:
        json.dump(model, f, indent=2)
    print("Guardado:", OUT_MODEL_JSON)

if __name__ == "__main__":
    main()