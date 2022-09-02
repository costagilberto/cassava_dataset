from pathlib import Path
from xml.etree import ElementTree

TOP_LEVEL = Path(__file__).parent.resolve()

DIRS = ["./low_abundance", "./moderate_abundance", "./super_abundance"]
CLASSES = ["whitefly"]

IMG_DIR = Path("images")
ANNOTATION_DIR = Path("annotation")
DEST_ANNOTATION_DIR = Path("annotation_txt")


def abs2frac_bndbox(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == "__main__":
    DIRS = [TOP_LEVEL.joinpath(set) for set in DIRS]

    xml_files = []
    jpg_files = []
    for dir in DIRS:
        print(f'-> {dir.resolve()}')
        # Fetch Annotations and Images
        jpg_files = list(dir.glob(str(IMG_DIR / "*.jpg")))
        xml_files = list(dir.glob(str(ANNOTATION_DIR / "*.xml")))

        # Assert for missing files
        diff_files = set([p.stem for p in xml_files]) - set([p.stem for p in jpg_files])
        assert not diff_files, f"Found missing XML or JPG for {diff_files}"

        for xml_file in xml_files:
            dir.joinpath(DEST_ANNOTATION_DIR).mkdir(exist_ok=True)
            with open(dir.joinpath(DEST_ANNOTATION_DIR, f"{xml_file.stem}.txt"), "wt") as outfile:
                tree = ElementTree.parse(xml_file)
                root = tree.getroot()
                size = root.find("size")
                w, h = [int(size.find(key).text) for key in ["width", "height"]]  # type: ignore

                for obj in root.iter("object"):
                    # Class and Class ID
                    cls = obj.find("name")
                    assert cls is not None
                    cls = cls.text
                    assert isinstance(cls, str)

                    """
                    Upon dataset inspection, some files present the 'whitefly' class as:
                        'whtefly'
                        'w'
                    """
                    if cls in ['whtefly', 'w']:
                        cls = CLASSES[0]
                    elif cls not in CLASSES:
                        print(f"DBG: found class {cls} in {xml_file}!")
                        continue
                    cls_id = CLASSES.index(cls)

                    # Fractional Bounding Box
                    xmlbox = obj.find("bndbox")
                    assert xmlbox is not None
                    box_labels = ["xmin", "xmax", "ymin", "ymax"]
                    b = tuple(float(xmlbox.find(bound).text) for bound in box_labels)  # type: ignore
                    bb = abs2frac_bndbox((w, h), b)

                    # Write to outfile
                    outfile.write(f'{cls_id} {" ".join(f"{b:.6f}" for b in bb)}\n')
