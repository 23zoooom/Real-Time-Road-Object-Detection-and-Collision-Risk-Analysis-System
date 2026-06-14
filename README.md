# Real-Time-Road-Object-Detection-and-Collision-Risk-Analysis-System
A real-time collision risk analysis system for front-camera driving video that converts frame-level road object perception into interpretable warning levels, combining object detection, multi-object tracking, region-of-interest (ROI) filtering, and a bounding-box height-based time-to-collision (TTC) proxy.

# Notice
The dataset cannot be provided due to license plate blurring issues.

# References
Vehicle Distance Measurement System:
https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System/blob/main/README.md

Ultralytics YOLO26
https://docs.ultralytics.com/models/yolo26#overview

ByteTrack 
https://github.com/FoundationVision/ByteTrack

# Contribution & Novelty
What I Referenced from Vehicle Distance Measurement System:
- ROI-based road-region design
- BBox height as a visual cue for distance / approach
- Warning visualization style

What I Added:
- ByteTrack-based object history across frames
- Shifted bbox height usage from single-frame distance estimation
 to temporal TTC estimation
- Smoothing for stable risk estimation
- HD Map / Autoware extension toward map-aware analysis
