# optimizers
Optimizers are algorithms or methods used to change the attributes of  machine learning/neural network such as weights and learning rate in order to reduce the losses.
This repo contains implementation of various optimizers with visualization.

## 1. SGD

### Algorithm:  
**θ=θ−α⋅∇J(θ)**

### Advantages:
* Frequent updates of model parameters hence, converges in less time.
* Requires less memory as no need to store values of loss functions.
### Disadvantages:
* High variance in model parameters.
* May shoot even after achieving global minima.

## 2. Momentum
It reduces high variance in SGD and softens the convergence. It accelerates the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction by accumularing past gradients. 

#### Algorithm:  
**V(t) = γV(t−1) + α.∇J(θ)  
θ    = θ − V(t)**
             
### Advantages:
* Reduces the high variance of the parameters.
* Converges faster than gradient descent.
### Disadvantages:
* One more hyper-parameter is added which needs to be selected manually and accurately.
* Overshooting

## 3. RMSProp
RMSProp also tries to dampen the oscillations, but in a different way than momentum. RMS prop also takes away the need to adjust learning rate, and does it automatically. More so, RMSProp choses a different learning rate for each parameter.

### Algorithm:  

**V(t)  = ρV(t−1) + (1 - ρ).∇J(θ)²  
∇W(t) =  -[α / (√V(t) + ϵ)].∇J(θ)  
θ     = θ + ∇W(t)**

### Advantages:
*  Reduces the oscillations.
*  Reduces overshooting.
### Disadvantages:
* One more hyper-parameter is added which needs to be selected manually and accurately.
* Slow convergence.
* Vanishing learning rate.

## 4. Adam
RMSProp and Momentum take contrasting approaches. While momentum accelerates our search in direction of minima, RMSProp impedes our search in direction of oscillations.
Adam or Adaptive Moment Optimization algorithms combines the heuristics of both Momentum and RMSProp.

 ### Algorithm:  
 **m(t) = β1 · m(t−1) + (1 − β1) · ∇J(θ)  
 v(t) = β2 · v(t−1) + (1 − β2) · ∇J(θ)²  
 θ     = θ - α . m(t) / (√V(t) + ϵ)**
  
### Advantages:
* Fast convergence.
* Rectifies vanishing learning rate, high variance.
* Reduces the oscillations.
* Reduces overshooting.
### Disadvantages:
* Computationally expensive.

Adam is the best optimizers. It trains neural networks in less time and more efficiently. But SGD can beat Adam in terms of accuracy if enough time is given(i.e. it takes too long).

## Installation/Usage

Clone the repository and open terminal in same directory and follow the below instuctions.
### Dependencies:
*  numpy
*  matplotlib
### To try codes:
#### SGD:
```python SGD.py```
![](https://github.com/girishdhegde/optimizers/blob/main/output/sgd.gif)
#### Momentum:
```python momentum.py```
![](https://github.com/girishdhegde/optimizers/blob/main/output/momentum.gif)
#### RMSProp:
```python RMSProp.py```
![](https://github.com/girishdhegde/optimizers/blob/main/output/rmsprop.gif)
#### Adam:
```python Adam.py```
![](https://github.com/girishdhegde/optimizers/blob/main/output/adam.gif)
#### All optimizers:
```python optimizers.py```
![](https://github.com/girishdhegde/optimizers/blob/main/output/optimizers.gif)
