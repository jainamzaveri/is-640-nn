import random
from engine import Value
from nn import MLP
data = [
    ([2.0, 3.0], 1.0),
    ([3.0, -1.0], -1.0),
    ([1.0, 1.0], 1.0),
    ([2.0, -2.0], -1.0)
]
model = MLP(2, [4, 1])
# Training loop
epochs = 100  # Number of iterations
learning_rate = 0.01

for k in range(epochs):
    # Forward pass: predict the output for each data point
    total_loss = Value(0)
    for x, y in data:
        x = [Value(xi) for xi in x]  # Convert inputs to Value objects
        y_pred = model(x)  # Forward pass
        loss = (y_pred - Value(y)) ** 2  # Mean squared error loss
        total_loss += loss
    
    model.zero_grad()  # Clear previous gradients
    total_loss.backward()  # Backpropagation
    
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    print(k, total_loss.data)
