import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader

data=load_breast_cancer()

x,y=data.data,data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.50,random_state=42,stratify=y_test)

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_val_scaled=scaler.transform(x_val)
x_test_scaled=scaler.transform(x_test)

print(x_train.shape)
print(x_val_scaled.shape)

x_train_tensor=torch.tensor(x_train_scaled,dtype=torch.float32)
x_val_tensor=torch.tensor(x_val_scaled,dtype=torch.float32)
x_test_tensor=torch.tensor(x_test_scaled,dtype=torch.float32)

y_train_tensor=torch.tensor(y_train,dtype=torch.float32).view(-1,1)
y_val_tensor=torch.tensor(y_val,dtype=torch.float32).view(-1,1)
y_test_tensor=torch.tensor(y_test,dtype=torch.float32).view(-1,1)

train_dataset=TensorDataset(x_train_tensor,y_train_tensor)
val_dataset=TensorDataset(x_val_tensor,y_val_tensor)
test_dataset=TensorDataset(x_test_tensor,y_test_tensor)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)

import torch
import os

class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.001):

        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_loss = float('inf')

        self.early_stop = False

        self.best_model_state = None

    def __call__(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.counter = 0

            # Save best model weights
            self.best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        else:

            self.counter += 1

            print(
                f"Early stopping counter: "
                f"{self.counter}/{self.patience}"
            )

        if self.counter >= self.patience:

            self.early_stop = True

    def restore_best_weights(self, model):

        if self.best_model_state is not None:

            model.load_state_dict(self.best_model_state)

            print("Best model weights restored.")

        else:

            print("No best model weights available.")




def save_checkpoint(model, optimizer, epoch):

    os.makedirs("models", exist_ok=True)

    checkpoint = {

        'epoch': epoch,

        'model_state_dict': model.state_dict(),

        'optimizer_state_dict': optimizer.state_dict()

    }

    torch.save(
        checkpoint,
        "models/latest_checkpoint.pth"
    )

    print(f"Checkpoint saved at epoch {epoch}")


early_stopping=EarlyStopping(patience=5, min_delta=0.001)

class CancerANN(nn.Module):

    def __init__(self,input_features):
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(input_features,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.network(x)


input_features=x_train.shape[-1]
model=CancerANN(input_features)

print(model)

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

epochs=100

training_loss=[]
val_losses=[]

try:
 for epoch in range(epochs):
    model.train()
    train_loss=0
    for x_batch ,y_batch in train_loader:
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    train_loss /=len(train_loader)
    training_loss.append(train_loss)

    model.eval()
    val_loss=0
    with torch.no_grad():
        for x_batch,y_batch in val_loader:
            outputs=model(x_batch)
            loss=criterion(outputs,y_batch)
            val_loss+=loss.item()
        val_loss/=len(val_loader)
        val_losses.append(val_loss)

        previous_best = early_stopping.best_loss
        
        early_stopping(val_loss, model)
        if early_stopping.best_loss < previous_best:
            os.makedirs("./model", exist_ok=True)
            torch.save(early_stopping.best_model_state,"./model/best_model.pth")
        print( f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )
        save_checkpoint(
            model,
            optimizer,
            epoch + 1
        )
        if early_stopping.early_stop:

            print("\nEarly stopping triggered!")

            break
except KeyboardInterrupt:

        print("\nTraining manually interrupted!")

        save_checkpoint(
        model,
        optimizer,
        epoch + 1
       )

finally:

    # Restore best validation model, if available
    early_stopping.restore_best_weights(model)

    print("\nTraining process finished.")


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,ConfusionMatrixDisplay

model.eval()

with torch.no_grad():
    outputs = model(x_test_tensor)

    probabilities = torch.sigmoid(outputs).numpy().ravel()

    predictions = (probabilities >= 0.5).astype(int)

actual = y_test

print("Accuracy:",
      accuracy_score(actual, predictions))

print("\nROC-AUC:",
      roc_auc_score(actual, probabilities))

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predictions,
        target_names=["Malignant", "Benign"]
    )
)

# Confusion matrix
cm = confusion_matrix(actual, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Malignant", "Benign"]
)
