### EX NO :8
### DATE  :
# <p align="center"> XOR GATE IMPLEMENTATION </p>
## Aim:
   To implement multi layer artificial neural network using back propagation algorithm.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:
Logic gates using neural networks help understand the mathematical computation by which a neural network processes its inputs to arrive at a certain output. This neural network will deal with the XOR logic problem. An XOR (exclusive OR gate) is a digital logic gate that gives a true output only when both its inputs differ from each other.

The information of a neural network is stored in the interconnections between the neurons i.e. the weights. A neural network learns by updating its weights according to a learning algorithm that helps it converge to the expected output. The learning algorithm is a principled way of changing the weights and biases based on the loss function.

## Algorithm
1.Import necessary packages

2.Set the four different states of the XOR gate

3.Set the four expected results in the same order

4.Get the accuracy

5.Train the model with training data.

6.Now test the model with testing data.


## Program:

Program to implement XOR Logic Gate.
Developed by   : mounika.s.c
RegisterNumber :  212219040084
```python3

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data =  np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")

model =Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())
```

## Output:
![exp8](https://user-images.githubusercontent.com/78891098/169592179-bfd5703c-6baf-4501-9582-1dc03bbc7af3.png)
![EXP 8](https://user-images.githubusercontent.com/78891098/169592199-1b563f56-5d98-4104-8fdf-99abaaf560d3.png)



## Result:
Thus the python program successully implemented XOR logic gate.
