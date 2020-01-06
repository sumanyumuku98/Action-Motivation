# Using Knowledge Graph to detect Intrinsic Motivation behind Actions from a given image. (Work-In-Progress)
#### Introduction
- The first part of the model is a simple `Encoder-Decoder` Network. The encoder is a pretrained `Resnet-101` which acts as a feature extractor.


![Resnet-101](https://www.jeremyjordan.me/content/images/2018/04/Screen-Shot-2018-04-16-at-6.30.05-PM.png)

- The decoder is a simple `GRU` cell to which encoded features are fed resulting in the decoded image-caption.

![Image-Captioning](https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png)

- The second part of the model uses NLP and graph exploration techniques to determine `motivation`. 

[ConceptNet](http://conceptnet.io) is an open source knowledge graph that represent the general knowledge involved in understanding language.

![ConceptNetGraph](https://i.dailymail.co.uk/i/pix/2015/10/08/09/2D31263700000578-3264560-ConceptNet_is_an_open_source_project_run_by_the_MIT_Common_Sense-a-9_1444294781970.jpg)

#### Method
  1. The resulting image caption is POS tagged to determine the `verbs` present, representing actions.
  
  2. A `keyword extractor` is used to extract significant words from image caption, which represents action's context correctly. 
  
  3. Using `ConceptNet API` the knowledge graph is explored to create a `subgraph`, consisting of all the edges(words) and vertices(semantic relation between words) that connects verbs to all the keywords.(*Hence the subgraph represents the entire context of action in the image through appropriate words*)
  
  4. Now using ConceptNet's `MotivatedByGoal` relationship of all the vertices in subgraph, The motivation behind the action is determined.
)
  



*ConceptNet is an active open project, whose knowledge is being regularly augmented through various sources, hence it will  only ameliorate our model's prediction of motivation.*   
