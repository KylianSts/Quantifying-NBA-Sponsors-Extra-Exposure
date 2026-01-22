# Computer Vision for Marketing Valuation: Quantifying NBA Sponsor Exposure

## Project Overview
This project operates at the intersection of computer vision and advertising analytics. The primary objective is to measure and monetize the value of unofficial NBA highlights uploaded by fans to YouTube. By using object detection models, this system transforms raw video footage into economic metrics to quantify the extra exposure sponsor gets from this videos.

![](Data/assets/demo.gif)

## Methodology

### Data Collection
 The system uses a hybrid data approach:
* **Visual Data:** A custom dataset of images extracted from YouTube highlights was created to train the vision model.
* **Tabular Data:** Tabular data including game dates, player statistics, and matchups was collected via the official NBA API to support predictive modeling.

### Visual Detection Architecture
The project utilizes the **YOLOv11-OBB (Oriented Bounding Box)** model. This architecture was selected for two reasons:
1.  **Geometry:** Standard bounding boxes cannot accurately trace tilted logos on a basketball court. Oriented boxes capture the rotation angle, ensuring precise segmentation.
2.  **Performance:** The model offers an optimal balance between accuracy (mean Average Precision) and processing speed, which is necessary for analyzing large volumes of video.

**Model Performance**
The selected YOLOv11-Small model achieved high precision results during evaluation:
* **Overall Accuracy:** The model reached a **mAP50 of 0.950** and a stricter **mAP50-95 of 0.867** across all classes.
* **Class Breakdown:** Static zones performed exceptionally well, with the "Back-Court Logo" achieving a score of **0.995**. The "Basketball" class, which is harder to track due to motion blur, achieved a score of **0.841**.
* **Configuration:** A confidence threshold of **0.6** was applied to prioritize certainty and minimize false positives, ensuring that brand exposure is not overestimated.

### Zone Taxonomy
The model detects six specific classes:
* **Mid_court_logo:** Central markings, often high-value assets.
* **Sid_court_logo:** Static markings near benches.
* **Sid_court_led_logo:** Dynamic LED panels on the sidelines.
* **Back_court_logo:** Markings along the baseline.
* **Basket_logo:** Small logos on the stanchion.
* **Basketball:** Used to track the focal point of viewer attention.

## Econometric Valuation

### The Quality Index
The project moves beyond simple exposure duration. It calculates a "Quality Index" for every detected logo based on four visual factors:
1.  **Visual Saliency:** The size of the logo relative to the screen.
2.  **Spatial Centrality:** The distance between the logo and the ball. The closer a logo is to the action, the higher its value.
3.  **Visual Clutter:** A measure of "Share of Voice." Value decreases if a logo is surrounded by many competing brands.
4.  **Signal Integrity:** The sharpness or blurriness of the logo.

### Financial Formula
The final monetary value is calculated by combining the reference Cost Per View (CPV), the duration of exposure, and the Quality Index.

## Predictive Modeling
To anticipate future value, the project tested three models: Linear Regression, Random Forest, and XGBoost. The models use historical data, such as team view counts and win percentages, to predict the media value of a matchup. XGBoost provided the best performance with an R-squared of 0.6622

## Key Results
* **Market Disconnect:** Social media value is seems to be driven by "Star Power" rather than just market size and team performance. Small markets with superstars can generate more value than large markets.
* **Zone Profitability:** The "Back-Court Logo" is the most valuable zone, generating over $4.3 million in the observed period (76 days) due to high visibility and low visual clutter.
* **Exposure:** Some teams, such as the Houston Rockets, had near-zero exposure in specific zones because their courts lack those specific branding assets.

## Limitations
* **Detection Method:** The current model uses frame-by-frame detection rather than object tracking, meaning it cannot identify if a logo appears in a continuous sequence.
* **Generic Classes:** The system detects zones (e.g., a generic LED panel) rather than identifying specific brand names[cite: 1388].
* **Data Volatility:** YouTube videos are frequently deleted, which complicates long-term historical analysis.