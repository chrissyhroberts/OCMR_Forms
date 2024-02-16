# OCMR_Forms

## Run cropandwarp.py on contents of input folder
* Detects the outer bounding box (frame) 
* Crops image to the outer bounding box
* Warps images to make the cropped region rectangular
* Trims away the outer bounding box to -p pixels
  
```python3 cropandwarp.py -i 01_input -o 02_cropped -p 50```

## Run prepare_template.py on a single file 
OMR/OCR system for reading data from paper checkbox forms.

options:
*  -h, --help            show this help message and exit
*  -i IMAGE_PATH, --image_path IMAGE_PATH - Path to the form image  
*  -w FIXED_WIDTH, --fixed_width FIXED_WIDTH  - Fixed width for cropping  
*  -ht FIXED_HEIGHT, --fixed_height FIXED_HEIGHT - Fixed height for cropping  
*  -t THRESHOLD, --threshold THRESHOLD - Threshold for checkbox detection  
*  -f FORM_NAME, --form_name FORM_NAME - Form name for naming output files  
*  -ma MIN_AREA, --min_area MIN_AREA - Minimum area of a checkbox  
*  -xa MAX_AREA, --max_area MAX_AREA - Maximum area of a checkbox  
*  -e EPSILON, --epsilon EPSILON - Epsilon value for contour approximation  
*  -s SIDES [SIDES ...], --sides SIDES [SIDES ...]  
       Number of sides for the contour approximation (e.g., 3, 4, 6 for triangles, rectangles, and hexagons).  
 ![test_10](https://github.com/chrissyhroberts/OCMR_Forms/assets/31275801/01c16338-dcaa-4dc1-a05a-e4c93921d59b)
      
```python3 prepare_template.py -i 02_cropped/test_10.png -ma 10 -xa 1000000 -e 0.01 -f template -s 3 5 8 ```
Creates a new annotated image

![test_10_annotated](https://github.com/chrissyhroberts/OCMR_Forms/assets/31275801/20d97758-4dc4-47ff-b21e-18628d36e5ca)


Creates a new file `form_a_template.json`

```
[
{"question": "Q_", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 1224, "centroid_y": 891, "area": 35823, "contour": [[1352, 857], [1225, 778], [1098, 857], [1147, 983], [1304, 982]]},
{"question": "Q_", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 806, "centroid_y": 649, "area": 3777, "contour": [[790, 619], [769, 637], [769, 662], [790, 680], [822, 680], [843, 662], [843, 636], [823, 619]]},
{"question": "Q_", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 1604, "centroid_y": 689, "area": 14278, "contour": [[1505, 737], [1704, 738], [1605, 594]]},
{"question": "Q_", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 1112, "centroid_y": 301, "area": 24800, "contour": [[1073, 222], [1019, 268], [1019, 334], [1074, 381], [1152, 381], [1207, 334], [1207, 268], [1152, 222]]},
{"question": "Q_", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 494, "centroid_y": 408, "area": 138524, "contour": [[401, 222], [272, 331], [272, 487], [402, 596], [590, 595], [718, 487], [718, 331], [589, 222]]}
]
```

At this stage you can edit the labels to match the questions in your form

```
[
{"question": "Q_01", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 1224, "centroid_y": 891, "area": 35823, "contour": [[1352, 857], [1225, 778], [1098, 857], [1147, 983], [1304, 982]]},
{"question": "Q_01", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 806, "centroid_y": 649, "area": 3777, "contour": [[790, 619], [769, 637], [769, 662], [790, 680], [822, 680], [843, 662], [843, 636], [823, 619]]},
{"question": "Q_02", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 1604, "centroid_y": 689, "area": 14278, "contour": [[1505, 737], [1704, 738], [1605, 594]]},
{"question": "Q_02", "subquestion": "b", "question_type": "select_multiple", "centroid_x": 1112, "centroid_y": 301, "area": 24800, "contour": [[1073, 222], [1019, 268], [1019, 334], [1074, 381], [1152, 381], [1207, 334], [1207, 268], [1152, 222]]},
{"question": "Q_03", "subquestion": "a", "question_type": "select_multiple", "centroid_x": 494, "centroid_y": 408, "area": 138524, "contour": [[401, 222], [272, 331], [272, 487], [402, 596], [590, 595], [718, 487], [718, 331], [589, 222]]}
]
```
```
