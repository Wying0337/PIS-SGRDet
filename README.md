# PIS-SGRDet

## Pseudo-image stitching and deep learning based method for large-scale corn seed germination rate detection


<p>
<img src=./images/Figure1.png align="center" width=1000 />
The flowchart of the proposed method.
  
</p>

## Environment


```python=3.8.20
numpy=1.20.0     
opencv-python=4.11.0.86    
matplotlib=3.7.5   
```

## The algorithm process is as follows: 
```
1. unit_sort.py is designed to restore the original arrangement of units within a local image. In this process, individual patch images from the image are considered distinct units, and the restoration is performed according to the detection results provided by a YOLO model.
python unit_sort.py
```

```
2. variant_temple_matching.py performs matching of patches from the same culture dish between two adjacent local images.
python variant_temple_matching.py
```

```
3. iterative_global_map.py iteratively synthesizes a global scene map to produce all culture dish patches for each coordinate location.
python iterative_global_map.py
```

```
4. optimize_patch_for_each_cor.py selects the culture dish patch from the optimal viewpoint as the final output.
pyhthon optimize_patch_for_each_cor.py
```
## The original and optimized results are displayed below:
![图片](./images/final_global_result_0.jpg "Original result")
