import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=load_breast_cancer()
x,y=data.data,data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

x_train_tensor=torch.tensor(x_train,dtype=torch.float32)
x_test_tensor=torch.tensor(x_test,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train,dtype=torch.float32).view(-1,1)
y_test_tensor=torch.tensor(y_test,dtype=torch.float32).view(-1,1)


class ANN(nn.Module):

    def __init__(self,input_features):
        super().__init__()
        self.layer1=nn.Linear(input_features,64)
        self.layer2=nn.Linear(64,32)
        self.output=nn.Linear(32,1)

    def forward(self,x):
        x=self.layer1(x)
        x=torch.relu(x)
        x=self.layer2(x)
        x=torch.relu(x)
        x=self.output(x)
        return x

model = ANN(input_features=x_train.shape[1])
criterion=nn.BCEWithLogitsLoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

epochs=100
loss_history=[]

for epoch in range(epochs):
    outputs=model(x_train_tensor)
    loss=criterion(outputs,y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch+1)%10==0:
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {loss.item():.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Batch Gradient Descent - Training Loss"
)

plt.show()

model.eval()
with torch.no_grad():
    test_logits=model(x_test_tensor)
    probabilities=torch.sigmoid(test_logits)
    predictions=(probabilities>=0.5).float()


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions.numpy())
print(f'test_accuracy:{accuracy:4f}')

from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions.numpy(),
        target_names=data.target_names
    )
)



