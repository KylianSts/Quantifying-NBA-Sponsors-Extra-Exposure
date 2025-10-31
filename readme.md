### **Project: Measuring Unofficial Sponsor Exposure in NBA Highlights**

#### **Project Vision**

This project aims to measure the on-screen presence of NBA team sponsors in unofficial highlight videos. Millions of fans who don't have subscriptions to official streaming services or the time to watch full games often turn to these unauthorized highlight reels on platforms like YouTube. This creates a significant amount of "extra" exposure for sponsors that is not (at least officialy) accounted for in sponsorship deals. By automatically analyzing these videos, we can provide teams and the NBA with valuable data to better understand and price sponsorships.

#### **Core Objective**

The main goal is to create a system that automatically collects NBA highlight videos and uses object detection to measure the screen time of specific **sponsorship zones**. Instead of training a model to find individual brand logos in itself, we will train it to recognize key areas like the center-court logo, the jersey patch, backcourt logos, and sideline ads. Brands want their logos in these zones, and our goal is to measure that exposure in non official (and non authorized highlights) to help set better prices.

We will combine this screen-time data with video metrics like view counts and likes to calculate an "unofficial exposure score." This will help NBA teams and the league get a clearer picture of sponsorship value, improve their pricing strategies, and even predict sponsor exposure for future games.

#### **Methodology**

Our process will start with collecting a large set of unofficial NBA highlight videos from the internet. A key part of the project will be building a custom object detection model, such as a fine-tuned YOLO model.

To train this model, we will use an efficient labeling method. We will begin by manually labeling the different exposure zones on a small set of a few hundred video frames. We will use this first set of images to train a basic model. This model will then be used to automatically suggest labels on a much larger set of new images. We will then manually check and correct these automated labels and use the new, larger dataset to retrain and improve the model. This loop helps create a large, high-quality dataset much faster. If labeling these zones becomes too time-consuming, we can simplify the approach and just detect the sponsor logos directly. This would be less detailed but still provide valuable information.

To save computing power and time, especially when analyzing a whole season of highlights, we will use a few smart strategies. We can use lower-quality video (for example, 480p resolution and 20 frames per second), which will speed up processing and reduce data storage needs without likely losing much accuracy (to test actually).

Additionally, instead of analyzing every single highlight video for a game, we can analyze a smaller, representative sample. From this sample, we can calculate an average exposure time for each zone, perhaps scaled to a standard length like ten minutes. This average can then be used to estimate the total exposure across all highlights for that game, saving a lot of time.

Looking forward, we plan to build a model to predict future sponsor exposure. We will add more information to our dataset from an official NBA API. This will include factors like the presence of All-Star or MVP players, team rankings, the popularity of rookie players, and the importance of the game, such as playoff match. This will allow the project to move from just measuring exposure to predicting it.