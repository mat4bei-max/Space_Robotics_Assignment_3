# YOLO ROS version

- Repository: https://github.com/mgonzs13/yolo_ros
- Version: **4.7.1**
- Required commit: `c6d29ce65da555b125517db6ac53729320422b2b`

Use this commit rather than the latest repository version. This project uses
the Python/Ultralytics implementation, which loads the trained PyTorch `.pt`
weights directly. The v5 implementation uses C++ and ONNX Runtime instead;
using it would require exporting the weights to `.onnx` and adapting the integration.
