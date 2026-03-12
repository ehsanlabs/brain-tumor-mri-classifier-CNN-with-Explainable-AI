import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, cv2, tensorflow as tf
from tensorflow import keras
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── SIRF YEH CHANGE KARO ─────────────────────────────────────────────────────
IMAGE_NUMBERS = [1447, 1451, 1455, 1458]   # <-- yahan numbers daalo
CLASS_FOLDER  = "glioma"                   # glioma / meningioma / notumor / pituitary
# ──────────────────────────────────────────────────────────────────────────────

FILE_PREFIX = {
    "glioma":     "gl",
    "meningioma": "m",
    "notumor":    "no",
    "pituitary":  "p",
}

CLASSES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = 128
BASE_DIR = r"D:\.FyP New Vesion"

prefix = FILE_PREFIX[CLASS_FOLDER]
images = [os.path.join(BASE_DIR, "dataset", CLASS_FOLDER, f"{prefix} ({n}).jpg")
          for n in IMAGE_NUMBERS]

# Check files exist
for p in images:
    if not os.path.exists(p):
        print(f"NOT FOUND: {p}")
        exit(1)

print(f"Testing {len(images)} images with guaranteed_90plus.keras ...")

model      = keras.models.load_model(os.path.join(BASE_DIR, "guaranteed_90plus.keras"))
grad_model = keras.Model(inputs=model.inputs,
                         outputs=[model.get_layer("last_conv").output, model.output])

rows = len(images)
fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
fig.patch.set_facecolor("#0d0d0d")
if rows == 1:
    axes = [axes]

for row, img_path in enumerate(images):
    fname    = os.path.basename(img_path)
    bgr_orig = cv2.imread(img_path)
    bgr128   = cv2.resize(bgr_orig, (IMG_SIZE, IMG_SIZE))
    rgb128   = cv2.cvtColor(bgr128, cv2.COLOR_BGR2RGB)
    arr      = np.expand_dims(rgb128.astype("float32") / 255.0, 0)

    # GradCAM
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.cast(arr, tf.float32))
        pred_idx = int(np.argmax(preds[0].numpy()))
        loss = preds[:, pred_idx]
    grads   = tape.gradient(loss, conv_out)[0].numpy()
    weights = np.mean(grads, axis=(0, 1))
    cam     = np.sum(weights * conv_out[0].numpy(), axis=-1)
    cam     = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam_up  = cv2.resize(cam.astype("float32"), (IMG_SIZE, IMG_SIZE))
    conf    = float(preds[0].numpy()[pred_idx]) * 100

    # Tumor mask
    gradcam_mask = (cam_up > 0.35).astype(np.uint8) * 255
    gray    = cv2.cvtColor(bgr128, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    _, otsu = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_and(otsu, gradcam_mask)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k2)
    combined = cv2.dilate(combined, k1, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    if n > 1:
        largest  = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        combined = ((labels == largest) * 255).astype(np.uint8)

    SIZE    = 320
    disp    = np.array(cv2.cvtColor(cv2.resize(bgr_orig, (SIZE, SIZE)), cv2.COLOR_BGR2RGB))
    cam320  = cv2.resize(cam_up, (SIZE, SIZE))
    heat    = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam320), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    cam_viz = cv2.addWeighted(disp, 0.4, heat, 0.6, 0)
    mask320 = cv2.resize(combined, (SIZE, SIZE))
    fill    = disp.copy(); fill[mask320 > 0] = [220, 50, 50]
    final   = cv2.addWeighted(disp, 0.5, fill, 0.5, 0)
    cnts, _ = cv2.findContours(mask320, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final, cnts, -1, (0, 255, 80), 2)

    correct = CLASSES[pred_idx].lower().replace(" ", "") == CLASS_FOLDER.replace(" ", "")
    result_color = "#00ff88" if correct else "#ff4444"

    for col, img, title in zip([0, 1, 2], [disp, cam_viz, final], [
        f"{fname}", "GradCAM Heatmap", f"{CLASSES[pred_idx]}  {conf:.1f}%"
    ]):
        ax = axes[row][col]
        ax.imshow(img)
        tc = result_color if col == 2 else "white"
        ax.set_title(title, color=tc, fontsize=11, pad=6)
        ax.axis("off")

    print(f"  {fname}  ->  {CLASSES[pred_idx]}  {conf:.1f}%  {'✓' if correct else '✗'}")

out = os.path.join(BASE_DIR, "test_outputs", "custom_test_result.png")
plt.tight_layout(pad=1.5)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"\nSaved: {out}")
