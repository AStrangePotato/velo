[![Watch the demo](http://img.youtube.com/vi/aR3xyo-Y60g/0.jpg)](https://www.youtube.com/shorts/aR3xyo-Y60g "Demo")

This project analyzes badminton smash technique improvement using computer vision. It uses MediaPipe Pose to extract and track human body landmarks from two videos, a player’s attempt and a reference model. The script normalizes body positions by centering on the hips and scaling by torso height, then compares joint movements frame-by-frame to compute average and per-joint deviations.

It aligns both videos using the loudest sound (from shuttle impact) for timing synchronization, then visualizes the comparison side by side. Each joint and limb is color-coded by deviation intensity (green = accurate, red = off), providing a visual and quantitative assessment of movement accuracy and form consistency.
