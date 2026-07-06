AI Virtual Product Placement (VPP) Pipeline 🎬🤖

An experimental computer vision pipeline designed for Dynamic In-Content Advertising. This project demonstrates how to seamlessly insert 2D product assets (like a soda can or branded object) into existing 2D video footage using modern AI tracking and segmentation, mimicking the technology used by streaming giants for personalized ad insertion.

🌟 The Vision

Imagine watching a movie on a streaming platform. In a coffee shop scene, User A sees the protagonist drinking a Pepsi. User B, watching the exact same scene, sees a Coca-Cola. By dynamically compositing products into scenes after filming, platforms can serve personalized, non-intrusive advertisements. This pipeline is a proof-of-concept for that technology.

⚙️ Features

This pipeline goes beyond simple video overlays by utilizing a multi-layered VFX approach:

Zero-Shot Camera Tracking: Uses Meta's CoTracker3 to calculate dense optical flow, locking the digital asset to the physical geometry of the scene without manual keyframing.

Dynamic AI Rotoscoping (Occlusion): Integrates YOLOv8 instance segmentation to extract the polygon boundaries of foreground objects (people, cars).

Z-Depth Sorting: Calculates the Y-axis floor-contact points of YOLO polygons to determine if an actor is walking in front of or behind the inserted product, masking it accordingly.

Environmental Harmonization: Uses OpenCV to calculate the standard deviation and mean color of the background pixels, dynamically adjusting the contrast, grain, and lighting of the inserted asset to match the scene's ambient environment.

🛠️ Tech Stack

Tracking: CoTracker3 (PyTorch)

Segmentation: YOLOv8 (Ultralytics)

Compositing & Harmonization: OpenCV, NumPy

Encoding: FFmpeg (for lossless frame stitching)


🚀 How to Run (Google Colab)

This project is optimized to run on a Google Colab T4 GPU instance to manage VRAM effectively.

Open a new Google Colab notebook (Runtime -> T4 GPU).

Upload your background video (Trial.mp4) and transparent product image (object.png).

Run the installation dependencies.

Set your target bounding box coordinates in the VFX Dashboard.

Execute the pipeline to generate your HD composited video.


🧠 Future Roadmap

Implementing DepthAnythingV2 for true 3D spatial awareness and pixel-perfect partial occlusion.

Adding Stable Diffusion Img2Img for automatic AI relighting of the product based on scene lighting direction.

Upgrading to a web-based GUI for easy bounding-box selection.
