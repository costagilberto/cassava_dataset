from pathlib import Path
from xml.etree import ElementTree
import shutil

import numpy as np
from numpy.random import default_rng

TOP_LEVEL = Path(__file__).parent.resolve()

DIRS = ["./low_abundance", "./moderate_abundance", "./super_abundance"]

IMG_DIR = Path("images")
TXT_ANNOTATION_DIR = Path("annotation_txt")

OUT_DIR = Path("cassava_dataset")
SYMLINK = False # if not symlink, copy files

TRAIN_SET_SIZE = 2400
VAL_SET_SIZE = 300
TEST_SET_SIZE = 300
N = TRAIN_SET_SIZE + TEST_SET_SIZE + VAL_SET_SIZE

RNG_SEED = 333

if __name__ == "__main__":
    DIRS = [TOP_LEVEL.joinpath(set) for set in DIRS]

    txt_files = []
    jpg_files = []
    for dir in DIRS:
        print(f"-> {dir}")
        # Fetch Annotations and Images
        jpg_files += list(dir.glob(str(IMG_DIR / "*.jpg")))
        txt_files += list(dir.glob(str(TXT_ANNOTATION_DIR / "*.txt")))

    # Assert for missing files
    diff_files = set([p.stem for p in txt_files]) - set([p.stem for p in jpg_files])
    assert (
        not diff_files
    ), f"Execute Conversion Script (voc2yolo_cassava_whitefly.py) to generate TXT annotations!!!"

    ## RNG
    rng = default_rng(RNG_SEED)

    # Dataset Partitioning
    set_idx = set(range(N))
    train_idx = set(rng.choice(list(set_idx), size=TRAIN_SET_SIZE, replace=False, shuffle=True))
    set_idx = set_idx - train_idx
    val_idx = set(rng.choice(list(set_idx), size=VAL_SET_SIZE, replace=False, shuffle=True))
    set_idx = set_idx - val_idx
    test_idx = set_idx
    
    # Dataset folder structure
    TOP_DIR = TOP_LEVEL.joinpath(OUT_DIR)
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    for (cat, idxs) in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
        TOP_DIR.joinpath('images', cat).mkdir(parents=True, exist_ok=True)
        #TOP_DIR.joinpath('labels', cat).mkdir(parents=True, exist_ok=True)
        
        # Symlink Images & Labels
        with open(TOP_DIR.joinpath(f"{cat}.txt"), "w+") as f:
            for idx in idxs:
                jpg_dataset_file = OUT_DIR.joinpath('images', cat, jpg_files[idx].name)
                f.write(f"{jpg_dataset_file}\n")
                
                jpg_file = TOP_DIR.joinpath('images', cat, jpg_files[idx].name)
                if SYMLINK:
                  try:
                      jpg_file.symlink_to(jpg_files[idx].resolve())
                      # print(f'{jpg_file} -> {jpg_files[idx].resolve()}')
                  except FileExistsError:
                      pass
                else:
                  shutil.move(jpg_files[idx].resolve(), jpg_file)



                txt_file = TOP_DIR.joinpath('labels', cat, txt_files[idx].name)
                if SYMLINK:
                  try:
                      txt_file.symlink_to(txt_files[idx].resolve())
                      # print(f'{txt_file} -> {txt_files[idx].resolve()}')
                  except FileExistsError:
                      pass
                else:
                  shutil.move(str(txt_files[idx].resolve()), str(txt_file))
