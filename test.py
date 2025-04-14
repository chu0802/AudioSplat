from pathlib import Path
import numpy as np
from PIL import Image
from pathlib import Path

root_dir = Path("/work/chu980802/langsplat/car/language_features")

colors = np.random.randint(0, 255, (300, 3)).astype(np.uint8)

for file in sorted(root_dir.glob("*_s.npy")):
    segmentation_map = np.load(file)[-1]
    file_name = file.stem
    print(file_name)
    


    max_idx = int(segmentation_map.max())
    min_idx = int(segmentation_map[segmentation_map != -1].min())
    
    num_classes = max_idx - min_idx + 1
    print(max_idx, min_idx, num_classes)
    # create a new segmentation map
    segmentation_map_new = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)

    # for each class, assign a random color
    for i in range(num_classes):
        segmentation_map_new[segmentation_map == i + min_idx] = colors[i]
        
    # for value -1, assign a black color
    segmentation_map_new[segmentation_map == -1] = np.array([0, 0, 0], dtype=np.uint8)

    print(segmentation_map_new.shape)

    Image.fromarray(segmentation_map_new.astype(np.uint8)).convert("RGB").save(f"segmentation_map_{file_name}.png")

