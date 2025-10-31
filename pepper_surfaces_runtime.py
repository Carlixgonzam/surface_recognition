

import time, threading, json
import numpy as np
import cv2
from naoqi import ALProxy
from skimage.feature import local_binary_pattern

PEPPER_IP = ""
PORT = 9559
CLIENTE = "surf_vision_tick"
CAMARA_INDEX = 0        
RESOLUCION = 2         
COLOR_SPACE = 11     
FPS = 10
TICK_SEC = 0.12         
ANUNCIO_COOLDOWN = 1.5
MODEL_JSON = "centroides_superficies.json"
RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = 'uniform'
HSV_BINS = (8, 8, 8)
DSCALE = 0.5
def cargar_modelo(path):
    with open(path, "r") as f:
        m = json.load(f)
    classes = m["classes"]
    centroids = {k: np.array(v, dtype=np.float32) for k, v in m["centroids"].items()}
    params = m.get("params", {})
    return classes, centroids, params

CLASSES, CENTROIDS, PARAMS = cargar_modelo(MODEL_JSON)
HSV_BINS = tuple(PARAMS.get("HSV_BINS", HSV_BINS))
RADIUS = int(PARAMS.get("RADIUS", RADIUS))
N_POINTS = int(PARAMS.get("N_POINTS", N_POINTS))
METHOD = str(PARAMS.get("METHOD", METHOD))
DSCALE = float(PARAMS.get("DSCALE", DSCALE))

def features_color_hsv(bgr):
    if DSCALE != 1.0:
        bgr = cv2.resize(bgr, (0,0), fx=DSCALE, fy=DSCALE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None, HSV_BINS, [0,180, 0,256, 0,256]).flatten().astype(np.float32)
    s = hist.sum()
    if s > 0: hist /= s
    return hist

def features_lbp(bgr):
    if DSCALE != 1.0:
        bgr = cv2.resize(bgr, (0,0), fx=DSCALE, fy=DSCALE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method=METHOD)
    bins = N_POINTS + 2 if METHOD == 'uniform' else int(lbp.max()+1)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0: hist /= s
    return hist

def extract_feature(bgr):
    h = features_color_hsv(bgr)
    t = features_lbp(bgr)
    feat = np.concatenate([h, t]).astype(np.float32)
    n = np.linalg.norm(feat)
    if n > 0: feat /= n
    return feat

def nearest_centroid(feat):
    best_c, best_d = None, 1e9
    for label, c in CENTROIDS.items():
        d = np.linalg.norm(feat - c)
        if d < best_d:
            best_d = d
            best_c = label
    return best_c, best_d

class PepperSurfaceClassifier(object):
    def __init__(self):
        self.cam = ALProxy("ALVideoDevice", PEPPER_IP, PORT)
        self.tts = ALProxy("ALTextToSpeech", PEPPER_IP, PORT)
        self.motion = ALProxy("ALMotion", PEPPER_IP, PORT)
        self.sub = self.cam.subscribe(CLIENTE, CAMARA_INDEX, RESOLUCION, COLOR_SPACE, FPS)
        print("OK: conectado a cámara.")
        self.tts.post.say("Clasificador de superficies activo.")
        self._stop = threading.Event()
        self._timer = None
        self.last_label = None
        self.last_announce = 0.0
        self.actions = {
            "pasto":   ("Pasto detectado. Avanzo con cuidado.", 0.15),
            "madera":  ("Madera detectada. Avance normal.", 0.30),
            "baldosa": ("Baldosa detectada. Avance normal.", 0.30),
            "alfombra":("Alfombra detectada. Avanzo lento.", 0.12),
        }

    def _tick(self):
        if self._stop.is_set():
            return

        img = self.cam.getImageRemote(self.sub)
        if img is not None:
            w, h, data = img[0], img[1], img[6]
            np_img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            feat = extract_feature(bgr)
            label, dist = nearest_centroid(feat)
            if label is None:
                label = "desconocida"
            now = time.time()
            if label != self.last_label and (now - self.last_announce) > ANUNCIO_COOLDOWN:
                msg, v = self.actions.get(label, ("Superficie desconocida. Reduzco velocidad.", 0.10))
                self.tts.post.say(msg)
                self.motion.post.setWalkTargetVelocity(v, 0.0, 0.0)
                self.last_announce = now
                self.last_label = label
            cv2.putText(bgr, "Clase: %s  (d=%.3f)" % (label, dist), (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Pepper - Superficie", bgr)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            self.stop()
            return

        self._timer = threading.Timer(TICK_SEC, self._tick)
        self._timer.start()

    def start(self):
        self._stop.clear()
        self._timer = threading.Timer(TICK_SEC, self._tick)
        self._timer.start()

    def stop(self):
        if not self._stop.is_set():
            print("Cierre solicitado…")
        self._stop.set()
        try: self.motion.stopMove()
        except: pass
        try: self.cam.unsubscribe(self.sub)
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        if self._timer is not None:
            try: self._timer.cancel()
            except: pass
        print("Limpieza completa.")

def main():
    clf = PepperSurfaceClassifier()
    clf.start()
    try:
        # Mantener vivo el hilo principal sin while True de procesamiento
        while not clf._stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupción manual.")
        clf.stop()

if __name__ == "__main__":
    main()