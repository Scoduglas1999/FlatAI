import os
import time
import numpy as np
import torch
from astropy.io import fits

from unet_model import AttentionResUNet

try:
    import xisf
except ImportError:
    xisf = None

try:
    from PIL import Image
except ImportError:
    Image = None


# Prefer final model if present; fall back to checkpoint
PREFERRED_MODEL = "./unet_flat_model_final.pth"
FALLBACK_MODEL = "./unet_flat_checkpoint.pth"
IMAGE_PATH_TO_CORRECT = "./Image34.fit"
CORRECTED_IMAGE_SAVE_PATH = "./flat_corrected.fits"

TILE_SIZE = 256
TILE_OVERLAP = 48
TILE_BATCH = 8  # tiles per forward pass
USE_AUTOCAST = True  # enable AMP on CUDA for speed


def pick_model_path():
    if os.path.exists(PREFERRED_MODEL):
        return PREFERRED_MODEL
    if os.path.exists(FALLBACK_MODEL):
        return FALLBACK_MODEL
    raise FileNotFoundError("No flat model file found. Expected one of: "
                            f"{PREFERRED_MODEL} or {FALLBACK_MODEL}")


def load_trained_unet(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = AttentionResUNet()
    checkpoint = torch.load(model_path, map_location=device)
    # Support both checkpoint dicts and raw state_dict saves
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_astro_image(image_path):
    if not os.path.exists(image_path):
        return None
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    try:
        if ext in ['.fits', '.fit']:
            with fits.open(image_path) as hdul:
                data = hdul[0].data
        elif ext == '.xisf':
            if xisf is None:
                raise ImportError("XISF provided but package not installed")
            data = xisf.read(image_path)[0].data
        elif ext in ['.jpg', '.jpeg', '.png']:
            if Image is None:
                raise ImportError("Pillow required for JPG/PNG")
            with Image.open(image_path) as img:
                data = np.array(img.convert('L'))
        else:
            return None
        if data.ndim == 3:
            data = data.mean(axis=2)
        return data.astype(np.float32)
    except Exception:
        return None


def preprocess_tile(tile_np):
    tmin, tmax = tile_np.min(), tile_np.max()
    if tmax > tmin:
        t01 = (tile_np - tmin) / (tmax - tmin)
    else:
        t01 = tile_np * 0.0
    t = t01 * 2.0 - 1.0
    t = torch.from_numpy(t.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return t


def postprocess_tile(tensor):
    x = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def process_single(image_2d, model, device):
    step = TILE_SIZE - TILE_OVERLAP
    window = np.hanning(TILE_SIZE)[:, None] * np.hanning(TILE_SIZE)[None, :]
    out = np.zeros_like(image_2d, dtype=np.float32)
    wmap = np.zeros_like(image_2d, dtype=np.float32)
    xs = list(range(0, image_2d.shape[1] - TILE_SIZE + 1, step))
    ys = list(range(0, image_2d.shape[0] - TILE_SIZE + 1, step))
    if xs[-1] < image_2d.shape[1] - TILE_SIZE:
        xs.append(image_2d.shape[1] - TILE_SIZE)
    if ys[-1] < image_2d.shape[0] - TILE_SIZE:
        ys.append(image_2d.shape[0] - TILE_SIZE)
    # Prepare all coordinates
    coords = [(y, x) for y in ys for x in xs]
    # Batch over tiles for speed
    idx = 0
    while idx < len(coords):
        batch_coords = coords[idx:idx + TILE_BATCH]
        tiles = []
        for (y, x) in batch_coords:
            tile = image_2d[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.std() < 1e-8:
                tiles.append(None)
            else:
                tiles.append(preprocess_tile(tile))

        preds = []
        if any(t is not None for t in tiles):
            # Find a reference tensor for zeros_like
            ref = next(t for t in tiles if t is not None)
            batch = torch.cat([t if t is not None else torch.zeros_like(ref) for t in tiles], dim=0).to(device)
            with torch.no_grad():
                if USE_AUTOCAST and device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        p = model(batch)
                else:
                    p = model(batch)
            # Split back
            preds = [p[i:i+1] for i in range(p.shape[0])]

        # Accumulate
        for i, (y, x) in enumerate(batch_coords):
            if tiles[i] is None:
                corrected = image_2d[y:y+TILE_SIZE, x:x+TILE_SIZE].copy()
            else:
                corrected = postprocess_tile(preds[i])
            out[y:y+TILE_SIZE, x:x+TILE_SIZE] += corrected * window
            wmap[y:y+TILE_SIZE, x:x+TILE_SIZE] += window
        idx += TILE_BATCH
    wmap[wmap == 0] = 1.0
    return out / wmap


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = pick_model_path()
    print(f"Loading model: {model_path}")
    model = load_trained_unet(model_path, device)
    img = load_astro_image(IMAGE_PATH_TO_CORRECT)
    if img is None:
        print("Unable to load image")
        return
    start = time.time()
    result = process_single(img, model, device)
    elapsed = time.time() - start
    print(f"Processed in {elapsed:.2f}s")
    # Save
    try:
        orig_hdr = None
        if IMAGE_PATH_TO_CORRECT.lower().endswith(('.fits', '.fit')):
            with fits.open(IMAGE_PATH_TO_CORRECT) as hdul:
                orig_hdr = hdul[0].header
        fits.writeto(CORRECTED_IMAGE_SAVE_PATH, result.astype(np.float32), header=orig_hdr, overwrite=True)
        print(f"Saved: {CORRECTED_IMAGE_SAVE_PATH}")
    except Exception as exc:
        print(f"Save failed: {exc}")


if __name__ == '__main__':
    main()


