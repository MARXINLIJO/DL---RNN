# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.
### Train Dataset
<img width="495" height="592" alt="image" src="https://github.com/user-attachments/assets/a3219a8e-f343-4e58-b201-c4c3a27eda84" />

### Test Dataset
<img width="365" height="590" alt="image" src="https://github.com/user-attachments/assets/da87df83-73bf-44ef-9612-d00ca920a98d" />

## DESIGN STEPS
### STEP 1: 
Load and normalize data, create sequences.
### STEP 2: 
Convert data to tensors and set up DataLoader.
### STEP 3: 
Define the RNN model architecture
### STEP 4: 
Summarize, compile with loss and optimizer.
### STEP 5: 
Train the model with loss tracking.
### STEP 6: 
Predict on test data, plot actual vs. predicted prices.
## PROGRAM
### Name: Marxin Lijo M
### Register Number: 212223240085
```python
## Define RNN Model
class RNNModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel,self).__init__()
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out
# Train the Model
# Write your code here
def train_model(model,train_loader,criterion,optimizer,epochs=20):
    train_losses=[]
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch,y_batch in train_loader:
            x_batch,y_batch=x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs,y_batch)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        train_losses.append(total_loss/len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    # Plot training loss
    print('Name:Marxin Lijo M')
    print('Register Number: 212223240085')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="595" height="479" alt="image" src="https://github.com/user-attachments/assets/8d1b39b9-6127-46b8-8bb7-81a43df063d8" />

## True Stock Price, Predicted Stock Price vs time
<img width="819" height="544" alt="image" src="https://github.com/user-attachments/assets/df82d484-8c33-4b87-a410-1df5bea90f07" />

### Predictions
<img width="227" height="35" alt="image" src="https://github.com/user-attachments/assets/baccc265-5a56-4f02-b8fa-d7985b9b15b0" />

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
