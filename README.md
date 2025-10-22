# BiteSense: Earable-Based Inertial Sensing for Eating Behaviour Assessment

BiteSense is a novel system that leverages earable inertial sensors (IMU) to monitor and analyze eating behavior. It detects chewing episodes, classifies food types using a hierarchical transformer-based model, and quantifies food intake, providing valuable insights for dietary assessment and health monitoring.

## Overview

Eating behavior analysis is essential for understanding dietary habits, managing chronic conditions, and promoting healthier lifestyles. BiteSense uses IMU data collected from earable devices (such as AirPods Pro) to:

- **Detect Eating Episodes:** Identify chewing cycles by processing 6-axis IMU data.
- **Hierarchical Food Classification:** Distinguish between various food types (e.g., solid, semi-liquid, liquid; crunchy, soft) through a multi-level classification approach.
- **Quantify Food Intake:** Estimate bite count, chewing intensity, and overall meal duration.

This repository includes the code for data preprocessing, feature engineering, model training, and evaluation as described in our research paper.

## Demo Video

Watch the demo video on YouTube to see BiteSense in action:  
[![BiteSense Demo Video](https://img.youtube.com/vi/3eEjJal1DQ8/0.jpg)]([https://www.youtube.com/watch?v=3eEjJal1DQ8](https://www.youtube.com/watch?v=3eEjJal1DQ8))

## Features

- **Automated Chewing Detection:** Leverages gyroscopic energy signals to reliably detect chewing episodes.
- **Hierarchical Classification Framework:** Uses a Transformer-based architecture to classify food types across multiple levels (state, texture, nutritional value, cooking method).
- **Intake Behavior Analysis:** Calculates bite rate and meal duration to distinguish between normal eating, overeating, and undereating.
- **Robust to Variations:** Validated across diverse scenarios and a wide demographic range.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/BiteSense.git
   cd BiteSense
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Citation

If you find this project useful in your research, please consider citing our work:
> **BiteSense: Earable-Based Inertial Sensing for Eating Behaviour Assessment**  
> Garvit Chugh, Indrajeet Ghosh, Sandip Chakraborty, Suchetana Chakraborty.  
> (DOI Coming Soon)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.
