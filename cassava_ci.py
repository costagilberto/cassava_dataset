from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from utils import check_bbox, get_slice_bboxes, img2label_paths

DATASET_DIR = Path(".").parent
DATASET_DIRS = [
    #*map(DATASET_DIR.joinpath, ["low_abundance", "moderate_abundance", "super_abundance"])
    *map(DATASET_DIR.joinpath, ["low_abundance"])
]

OUT_DIR = DATASET_DIR.joinpath("CI")

IMG_DIRS = [set / "images" for set in DATASET_DIRS]
TXT_ANNOTATION_DIR = [set / "annotation_txt" for set in DATASET_DIRS]
XML_ANNOTATION_DIR = [set / "annotation" for set in DATASET_DIRS]

SLICE_HEIGHT = 128
SLICE_WIDTH = 128
SLICE_OV_HEIGHT = 0.0
SLICE_OV_WIDTH = 0.0


def process_single_img(file, label):
    # Load image
    img = cv2.imread(str(file))
    (h, w) = img.shape[:2]

    # Load Labels
    with open(label, "r") as f:
        l = [x.split() for x in f.read().strip().splitlines()]
        l = np.array(l, dtype=np.float32)

    # Drop Labels with no Width or Height
    drop_l = []
    for n, obj in enumerate(l):
        if not np.all(obj[2:]):
            print(f"{label.name} -- Dropped Label {n}/{len(l)}: {obj}")
            drop_l.append(n)
        else:
            check_bbox(obj[1:], xywh=True)
    l = np.delete(l, drop_l, axis=0)

    # Compute Slices
    slices = get_slice_bboxes(h, w, SLICE_HEIGHT, SLICE_WIDTH, SLICE_OV_HEIGHT, SLICE_OV_WIDTH)

    bbox_params = A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.5)
    composed_slices = [A.Compose([A.Crop(*slice)], bbox_params) for slice in slices]  # type: ignore
    tf_slices = map(lambda x: x(image=img, bboxes=l[:, 1:], labels=l[:, 0]), composed_slices)

    # Save to Output
    out_p = OUT_DIR / file.parent.relative_to(DATASET_DIR)
    for n, tf in enumerate(tf_slices):

        # Save transformed labels
        with open(img2label_paths(out_file), "w+") as lb:
            if tf["bboxes"]:
                for obj in tf["bboxes"]:
                    check_bbox(obj, xywh=True)
                    lb.write(f"0 {obj[0]:.6f} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f}\n")
                    
                    # Save image
                    out_file = out_p / f"{file.stem}_{n}.jpg"
                    cv2.imwrite(str(out_file), tf["image"])


if __name__ == "__main__":
    # mkdir
    CI_FOLDERS = IMG_DIRS + TXT_ANNOTATION_DIR
    for fd in CI_FOLDERS:
        OUT_DIR.joinpath(fd).mkdir(parents=True, exist_ok=True)

    # image files & labels
    img_files = []
    for img_p in IMG_DIRS:
        img_files += [*img_p.glob("*.jpg")]
    files = [f.stem for f in img_files]
    labels = [img2label_paths(img_p, check=True) for img_p in img_files]

    # Run Processing Jobs
    jobs = []
    pbar = tqdm(total=len(files), desc="Images")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit Jobs
        for (file, label) in zip(img_files, labels):
            future = executor.submit(process_single_img, file, label)
            future.add_done_callback(lambda _: pbar.update(1))
            jobs.append(future)

        # Wait....
        try:
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
        finally:
            pbar.close()
    results = [future.result() for future in jobs]
