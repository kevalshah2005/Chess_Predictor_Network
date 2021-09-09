# Chess Match Predictor

What if you could predict the results of a chess match without knowing the players' individual moves?

The main goal of our project was to create a neural network that could anticipate the outcome of a chess game, given only a small number of variables. More specifically, our group wanted to create a machine learning model that was more effective at predicting the results of chess matches than simpler methods, such as random guessing. Below is our report describing the process of how the neural network was created, the results the model yielded, and what conclusions can be drawn from this information.

## Main Sections of Report:
1. [Intro to Neural Networks](#introduction-to-neural-networks)
2. Our Data
3. Neural Network Overview
4. Results and Analysis
5. Future Work and Improvements
6. Applications
7. Sources and Acknowledgements

## Introduction to Neural Networks

Defined formally, neural networks are computer systems modeled on the human brain and nervous system. These algorithms employ machine learning in order to produce increasingly accurate outputs, based on inputted data sets. Before understanding this process, one must first understand the architecture of a neural network. As shown in Fig. 1, they are made up of connected layers of nodes, which work sequentially to manipulate the input in different ways. Each node represents some component of the inputted variables, the value of which changes throughout the learning process. Depending on what kind of layer they are in, the nodes have diffeerent purposes. For example, in basic sequential networks, there are three main types of layers: input, which takes the data in and applies weights (numerical values representing the variable's importance), hidden, which apply weights to values given by the previous layer and pass the generated value to the next layer, and output, which contain values that represent the network's prediction.

![image](https://user-images.githubusercontent.com/74797855/132502675-7d31f25d-f3b9-4298-9e34-b3f54ee53015.png)

Fig. 1 _Example of basic neural networks archiecture_

### Network Learning
In order to improve a neural network's accuracy, the structure described above repetitively produces outputs, each time changing the weights of each node. It makes these changes based on the backpropogation algorithm. This algorithm minimizes a network's loss, which represents the distance between the outputted value and the expected value. As this is done after each output is produced, the network becomes more increasingly accurate. This process is called training. During training, a network's accuracy is also tesed on validation data, in order to make sure the algorithm is applicable to new data, not just the training data. If a neural network is found to have lower validation accuracy than training accuracy, it is considered to be suffering from overfitting (displayed in Fig. 2). The solution to the issue is to decrease the number of epochs (repetitions) in training, as doing so makes the network more applicable to other data. When the optimal number of layers, nodes, and epochs is found, the network can then be used on new input data.

![image](https://user-images.githubusercontent.com/74797855/132511763-ae1f4c72-ffc2-4071-8601-448ab1bbd9ad.png)

Fig. 2 _Plot showing the effects of overfitting_

## Our Data

The original data set we used for our neural network contained information about 6.3 million chess games played on Lichess.org (part of which shown in Fig. 3), but that amount had to be decreased to about 1 million due to a lack of ram. Initially, this set contained several variables of each game, such as each player's username, the time at which each match was played, and the date of each game. Given that most of these details were irrelevant to our specific goal of predicting which player would win, the number of columns was eventually widdled down to 5.

![image](https://user-images.githubusercontent.com/74797855/132511841-55e7cf96-18c2-42c2-abfb-c6c2f5824135.png)

Fig. 3 _Exerpt from the original data set_

### Variables Used
The first variable in our input data set (Fig. 4) was the event, which represented which kind of chess game each match was. This variable is important, because the amount of time in a game has a signficant influence on each player's moves. There were 7 possibilities: blitz, blitz tournament, bullet, bullet tournament, classical, classical tournament, and correpsondence. The second and third columns contained the ELO of black and white, respectively. This variable is essentially a representation of a player's overall performance as a chess player. Each time a player wins or loses, their ELO increases or decreases based on their opponent's rating. For example, if one player's ELO is signficantly higher than the other's, that player will most likely win the game. But, given this large difference in rating, each player's ELO will only change depending on the level of disadvantage- you get more points for beating someone significantly better than you, but lose fewer if you lose to someone better. On the other hand, if two players are very evenly matched, their ELO will shift only slightly based on if they win or lose. The theoretical maximum of this rating system is 3,000, although no one is currently at that level; if they managed to approach it a single loss would tank their score. The fourth column in our data set contained the Encyclopedia of Chess Openings (ECO) code for each match. Each of these codes has a letter and two digits (A00-E99) which help to determine the specific variation of the opening. Finally, the fifth column included in our data set was the game termination, which was either normal (checkmate) or time forfeit. The combination of these variables gave the network an adequate amount of information to make accurate predictions, without simply knowing all of each player's moves in a match.

![image](https://user-images.githubusercontent.com/74797855/132512120-a4e9aafe-c698-4831-ac6a-d250073c6ee3.png)

Fig. 4 _Visual representation of input data set_

### Target Data Set
The target dataset was used to contain the expected outputs for each chess match. The three possible outcomes were black wins, white wins, and tie. Fig. 5 shows this data set in a table format.

![image](https://user-images.githubusercontent.com/74797855/132513194-833ac871-5405-4d61-84fe-5e861f7aff81.png)

Fig. 5 _Visual representation of target data set_

### Data Manipulation
In order to input our training data and target data into the neural network, they had to be converted from it's raw format into a normalized, network compatible structure. To do so, we one-hot encoded the data values using the Pandas function "get_dummies." This function changes each value in the data set from a single number into a binary array of possible numbers. More specifically, each column is split up into multiple columns of zeroes and ones, the amount of which is defined by the number of possible states of the variable. For example, since the event column has 7 possibilites, it is seperated into 7 columns each containing a 0 or a 1 (displayed in Fig. 6). If a specific chess match was of a certain type, the zero in that column would be changed to a one. This process normalizes the data, making it compatible with the neural network. Once we had completed one-hot encoding our data, it simply had to be converted into a numpy array and inputted into the network.

![image](https://user-images.githubusercontent.com/74797855/132558614-66bcfb76-e4f8-4d75-ba93-86337f03c4b9.png)

Fig. 6 _Table of one-hot encoded data set_

In order to ensure that our network was not suffering from overfitting, we split up the data and target arrays into training data/targets and validation and testing data/targets. Doing so allowed us to train the network using most of the data set, while testing the model on "new" data, to make sure it wasn't baised towards only one data set. As shown in the image below, we divided the first 800,000 data points into the training data and target arrays, and allotted the the rest of the data set (about 200,100 data points) into the validation/testing data and target arrays.

![image](https://user-images.githubusercontent.com/74797855/132559623-046b16d8-23b4-4686-b15b-627ec2f025ff.png)

_Seperation of data set into the training data/targets arrays and the validation/testing data/targets arrays_

## Neural Network Overview

### Architecture
The final build of our neural network was a dense six layer model. After the input layer (which had as many nodes as there were columns in the data set), our 4 hidden layers had 16, 16, 8, and 4 nodes resepectively (displayed visually in Fig. 7). Our output layer then contained three nodes, as the possible outcomes of each chess game are black wins, white wins, and tie. This network structure was effective for our project, because it forced the computer to not overcomplicate the weights of each variable during the prediction process. More specifically, the relatively small number of nodes in each layer decreased efficiently, making the value manipulation between the penultimate layer and the output layer simpler (which made the network's preditions more consistently accurate). 

![image](https://user-images.githubusercontent.com/74797855/132579660-455cf4fc-3983-4d9f-bcd2-688cd04482f9.png)

Fig. 7 _Visualization of neural network_

### Other Network Components

To make sure our model trained efficiently, our group chose network functions that were the most effective at performing their designated tasks specifically in the context of our project. For example, we chose to use the rectified linear activation function (in the input and hidden layers) over sigmoid activation, because it doesn't cause issues with vanishing gradients (when the gradient of a function is appoaching 0, so little to no learning occurs). For the output layer activation, the clear choice was softmax because it is specifically effective for multi-class prediction models, such as our network. The function works by assigning a decimal probability to each possible result, then outputing the most likely outcome. The loss function utilized by our network was cateogrical cross-entropy, which works by calculating the difference between two probability distributions. In the context of our project, this means the difference between the each of the model's predictions and the respective targets. The accuracy of the network was represented using a simple percentage, as our predictions were discrete values (as opposed to continuous outputs, such as percentages). Lastly, our network employed the RMSProp optimization algorithm for performin backpropogation. This algorithm is especially effective, because it normalizes gradients using a moving average of squared gradients in order to prevent exploding (overstepping) for large gradients and vanishing (understepping) for small gradients. The code reuqired to employ these functions in a network is displayed below.

![image](https://user-images.githubusercontent.com/74797855/132605407-804a2418-d0ab-4917-afd2-b732cc37ca23.png)

_Written implementation of loss function, layer activations, accuracy tracker, and optmizer_

## Results and Analysis

Our netwok was able to predict the outcome of a completely new chess match 62.08% of the time (after epoch 7), with a validation/testing loss of ~0.75. Figs. 8 and 9 display the evolution of these output accuracy up until epoch 7, as well as afterwards, when the network began to suffer from overfitting.

![image](https://user-images.githubusercontent.com/74797855/132607297-53125e56-ef89-4470-ac94-4d25272f4abe.png)

Fig. 8

![image](https://user-images.githubusercontent.com/74797855/132607344-eee9f6b3-83ea-4594-8381-d970ea029639.png)

Fig. 9

But, without context, the signficance of this percentage is difficult to interpret. This is why it's important to comapre it to other prediction methods. The simplest algorithm to do this with is random guessing, as its expected accuracy is easy to calculate. Given that there were three possible outputs, guessing the result of a chess game randomly would (theoretically) yield an accuracy of 33.33%. Our model clearly performed signficantly better than this, as its accuracy is almost double that of the compared strategy. Another relatively simple prediction algorithm is to choose the outcome of each match solely on the players' ELO (meaning the higher rated player always wins, unless they are equal in which case the match would end in a tie). Using this strategy on our data set yielded an accuracy of 58%, further representing the effectiveness of our model. Fig. 10 provides a visual representation of the accuracies of random guessing, ELO only prediction, and our network.

![image](https://user-images.githubusercontent.com/74797855/132607532-8730808b-7f34-401d-8d9c-5e5fb2cb8349.png)

Fig. 10

To summarize our findings, the results displayed by these calculations and comparisons demonstrate two main details about our data set and network. Firstly, the model overfitting at epoch 7 demonstrates that there were imperfections in our data set and or neural network. One possible culprit is that the data set simply wasn't large enough to prevent the model from becoming biased towards the training data. Another aspect of the project that could have caused this problem is that the network structure wasn't in its most effective form (meaning the amount of layers and nodes were not completely optimized). On a more positive note, these results show that our model was successful in learning the importance of variables included in the data set other than just ELO (one of the most significant factors in trying to predict a match's outcome).

## Future Work and Improvements

While our model worked well, there are still a few things that we want to improve on in the future. First of all, we want to be able to use more RAM. We tried to run all 6 million games we had collected (and a few million more that we had stored up in case we needed them), but we ran out of RAM after only a million games, even with 24 GB of RAM. If we are able to run this on a powerful computer with hundreds of gigabytes of RAM, the model will have more games to go off of and thus will be more accurate. Another thing that we would like to do in the future is optimising the layers and nodes in each layer a little better. Finding the optimal balance between these two would allow us to get the highest accuracy possible with our data and type of network. 

## Applications

Our model can be used to improve the openings of amateur chess players. They can look at the data and see which openings performed the best for the ELO range that they are in and implement them into their own chess game. Obviously, chess openings are determined by both players, but with enough practice they can steer the game somewhat in the direction that they want to give them an advantage. Additionally, spectators who are not very familiar with chess can use this tool to get an idea of which player is most likely to win after a few moves.

## Sources and Acknowledgements

### Libraries Used
-Numpy
-Tensorflow
-Pandas
-Keras
-Matplotlib

### Original Data set
Link: https://www.kaggle.com/arevel/chess-games

### Sigmoid vs. ReLU
Article: https://medium.com/geekculture/relu-vs-sigmoid-5de5ff756d93

### General NN Information
Book: https://www.manning.com/books/deep-learning-with-python-second-edition?gclid=CjwKCAjwvuGJBhB1EiwACU1AiWRYAeI7L8XWG6nEAfMYcXaUeK6hTgCq5X21Y5OjxIHy0a2Bf0Xa_hoCC84QAvD_BwE
