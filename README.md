# NMS_C++

## why this project

This is just 2D rotated nms part of my project for single classification.After searching NMS C++ on github, most of the NMS project only provide the cuda nms function, but we can't just pass the model's output to cuda nms function, we should first remove the low probability anchor, and then sorting the anchor.

This project give a simple example for removing, sorting,and then nms.
The project can't run directly, the thresing, sorting, and nms part is in demo/faf.cpp, in FAF::inference function.

