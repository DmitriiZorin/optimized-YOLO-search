# Optimized YOLO usage example

In this repo example usage of YOLO model to detect people in vifeostream is used. Also author is informed about other infrastructure optimizations (such as converting models to onnx format etc), in this repo algorithmical side will be discussed. 

Use model on all frames may be simple solution, but not the most effective. During capturing after first detection, searching area can be narrowed. In the OptimizedYOLOFinder class this method is implemented. 

Usage: 

```
python main.py <file with YOLO model> <input file> <output file>
```

In the output file red rectangle is the search area rectangle and by the green rectangle persons is outlined.

Currently only .mp4 files are supported.
