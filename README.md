# Using Knowledge Graph to detect Intrinsic Motivation behind Actions. (Work-In-Progress)
#### Introduction
- The first part of the model is a simple `Encoder-Decoder` Network. The encoder is a pretrained `Resnet-101` which acts as a feature extractor.

![Resnet-101](https://www.jeremyjordan.me/content/images/2018/04/Screen-Shot-2018-04-16-at-6.30.05-PM.png)

- The decoder is a simple `GRU` cell to which encoded features are fed and image-captioning is done.

![Image-Captioning](https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png)

- The second part of the model is more NLP related. We use a keyword extractor to extract keywords from the caption. The keywords are then used for searching in the Open-Source Knowledge Graph `ConeceptNet` for semantic relations. Finally relevant motivation is detected using the `ConceptNet API`.

![ConceptNet](https://i.dailymail.co.uk/i/pix/2015/10/08/09/2D31263700000578-3264560-ConceptNet_is_an_open_source_project_run_by_the_MIT_Common_Sense-a-9_1444294781970.jpg)
