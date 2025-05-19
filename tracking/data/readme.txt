[yolo-dataset.txt]

Each line in the annotation file for the segmentation dataset follows the format:
class_id x y w h

x, y represent the normalized center coordinates of the bounding box, relative to the image width and height.
w, h represent the normalized width and height of the bounding box, also relative to the input image size.

Multiple bounding boxes for a single image are separated by spaces on the same line.

All class_id values are 0, indicating beetles.


------


[resnet-dataset.txt]

class_id: label
0: H
1: O
2: X
3: nn
4: ss
5: xx