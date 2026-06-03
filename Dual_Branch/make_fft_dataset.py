from pathlib import Path
from shutil import copy2

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent

LIST_ROOT = PROJECT_DIR / "Dual_Branch" / "data" / "LEVIR_MCI"
SRC_ROOT = PROJECT_DIR / "data" / "coding" / "datasets" / "LEVIR-MCI-dataset" / "images"
DST_ROOT = PROJECT_DIR / "data" / "coding" / "datasets" / "LEVIR-MCI-dataset-fft" / "images"

SPLITS = ("train", "val", "test")
IMAGE_FOLDERS = ("A", "B")
LABEL_FOLDERS = ("label", "label_rgb")

LOW_FREQ_SUPPRESS_RADIUS_RATIO = 0.08
LOW_FREQ_SUPPRESS_STRENGTH = 1


def read_names(split):
    list_path = LIST_ROOT / f"{split}.txt"
    names = []
    with list_path.open("r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            names.append(name.split("-")[0])
    return names


def normalize_to_uint8(x):
    x = x.astype(np.float32)
    x = x - x.min()
    max_value = x.max()
    if max_value > 0:
        x = x / max_value
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def low_frequency_suppressed_image(image):
    image = np.asarray(image.convert("RGB"), dtype=np.float32)
    channels = []

    for channel_idx in range(image.shape[2]):
        channel = image[:, :, channel_idx]
        spectrum = np.fft.fft2(channel)
        spectrum = np.fft.fftshift(spectrum)

        h, w = channel.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * LOW_FREQ_SUPPRESS_RADIUS_RATIO)
        y, x = np.ogrid[:h, :w]
        low_freq_mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        spectrum[low_freq_mask] *= 1.0 - LOW_FREQ_SUPPRESS_STRENGTH

        restored = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
        channels.append(normalize_to_uint8(restored))

    return np.stack(channels, axis=2)


def save_fft_image(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        fft_arr = low_frequency_suppressed_image(image)
    Image.fromarray(fft_arr).save(dst_path)


def copy_label(src_path, dst_path):
    if not src_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(src_path, dst_path)


def main():
    print("Source:", SRC_ROOT)
    print("Destination:", DST_ROOT)

    for split in SPLITS:
        names = read_names(split)
        for name in tqdm(names, desc=f"fft {split}"):
            for folder in IMAGE_FOLDERS:
                src_path = SRC_ROOT / split / folder / name
                dst_path = DST_ROOT / split / folder / name
                if not src_path.exists():
                    raise FileNotFoundError(src_path)
                save_fft_image(src_path, dst_path)

            for folder in LABEL_FOLDERS:
                src_path = SRC_ROOT / split / folder / name
                dst_path = DST_ROOT / split / folder / name
                copy_label(src_path, dst_path)

    print("Done:", DST_ROOT)


if __name__ == "__main__":
    main()
