# Chess Match Predictor

What if you could predict the results of a chess match without knowing the players' individual moves?

The main goal of our project was to create a neural network that could anticipate the outcome of a chess game, given only a small number of variables. More specifically, our group wanted to create a machine learning model that was more effective at predicting the results of chess matches than simpler methods, such as random guessing. Below is our report describing the process of how the neural network was created, the results the model yielded, and what conclusions can be drawn from this information.

## Main Sections of Report:
1. Introduction to Neural Networks
2. Our Data
3. Neural Network Overview
4. Results and Analysis
5. Possible Improvements
6. Applications
7. Acknowledgements

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
The first variable in our input data set (Fig. 4) was the event, which represented which kind of chess game each match was. This variable is important, because the amount of time in a game has a signficant influence on each player's moves. There were 7 possibilities: blitz, blitz tournament, bullet, bullet tournament, classical, classical tournament, and correpsondence. The second and third columns contained the ELO of black and white, respectively. This variable is essentially a representation of a player's overall performance as a chess player. Each time a player wins or loses, their ELO increases or decreases based on their opponent's rating. For example, if one player's ELO is signficantly higher than the other's, that player will most likely win the game. But, given this large difference in rating, each player's ELO will only change a small amount. On the other hand, if two players are very evenly matched, their ELO will shift signficantly based on if they win or lose. The theoretical maximum of this rating system is 3,000, although no one is currently at that level. The fourth column in our data set contained the Encyclopedia of Chess Openings (ECO) code for each match. Each of these codes has a letter and two digits (A00-E99) which help to determine the specific variation of the opening. Finally, the fifth column included in our data set was the game termination, which was either normal (checkmate) or time forfeit. The combination of these variables gave the network an adequate amount of information to make accurate predictions, without simply knowing all of each player's moves in a match.

![image](https://user-images.githubusercontent.com/74797855/132512120-a4e9aafe-c698-4831-ac6a-d250073c6ee3.png)
Fig. 4 _Visual representation of input data set_

### Target Data Set
The target dataset was used to contain the expected outputs for each chess match. The three possible outcomes were black wins, white wins, and tie. Fig. 5 shows this data set in a table format.

![image](https://user-images.githubusercontent.com/74797855/132513194-833ac871-5405-4d61-84fe-5e861f7aff81.png)
Fig. 5 _Visual representation of target data set_

### Data Manipulation
**Bold** and _Italic_ and `Code` text
