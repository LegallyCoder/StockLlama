# StockLlama
![image](https://github.com/user-attachments/assets/11d12a8f-63b8-42ce-b66c-d77924831e90)


StockLlama is a time series forecasting model based on Llama, enhanced with custom embeddings for improved accuracy.

# Usage:
To use the **StockLlama**, follow these steps:

1. Clone the repository to your local machine.
   
```bash
git clone https://github.com/LegallyCoder/StockLlama
```
2. Open a terminal or command prompt and navigate to the script's directory.
```bash
cd src
```

3. Install the required packages using this command:

```bash
pip3 install -r requirements.txt
```

4. Open new python file at the script's directory.
```python
import yfinance as yf
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
from modeling_stockllama import StockLlamaForForecasting
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLlamaForForecasting.from_pretrained("StockLlama/StockLlama").to(device)
day = 365
def download_stock_data(stock_symbol):
    end_date = datetime.today().date()
    start_date = datetime.today().date() - timedelta(days=day)
    try:
        return yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Error downloading data for {stock_symbol}: {e}")
        return None

def predict_future_prices(stock_symbol):
    stock_data = download_stock_data(stock_symbol)
    if stock_data is not None:
        subset = stock_data[['Close']].tail(day).reset_index(drop=True)
        model.eval()

        def prepare_data(data):
            return torch.tensor(data.values, dtype=torch.float32).unsqueeze(0).to(device)

        data_tensor = prepare_data(subset)
        future_predictions = []

        with torch.no_grad():
            for _ in range(day):
                output = model(data_tensor.squeeze(-1)).logits

                if len(output.shape) == 3:
                    last_prediction = output[:, -1, :].squeeze(0)
                elif len(output.shape) == 2:
                    last_prediction = output.squeeze(0)
                else:
                    raise ValueError("Unexpected model output shape.")

                future_predictions.append(last_prediction.item())

                if len(output.shape) == 3:
                    data_tensor = torch.cat((data_tensor[:, 1:, :], output[:, -1, :].unsqueeze(1)), dim=1)
                elif len(output.shape) == 2:
                    data_tensor = torch.cat((data_tensor[:, 1:], last_prediction.unsqueeze(0).unsqueeze(0)), dim=1)
        future_predictions = gaussian_filter1d(future_predictions, sigma=1)
        combined_prices = pd.concat([subset['Close'], pd.Series(future_predictions)], ignore_index=True)
        historical_dates = stock_data.index[-day:].to_list()
        prediction_dates = [historical_dates[-1] + timedelta(days=i) for i in range(1, len(future_predictions) + 1)]
        combined_dates = historical_dates + prediction_dates

        plt.figure(figsize=(12, 6))
        plt.plot(combined_dates[:len(subset)], combined_prices[:len(subset)], label='Historical Prices', linestyle='-')
        plt.plot(combined_dates[len(subset)-1:], combined_prices[len(subset)-1:], label='Predicted Prices', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{stock_symbol} - Combined Historical and Predicted Prices')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return future_predictions
    else:
        print(f"Data could not be downloaded for {stock_symbol}.")
        return None

stock_symbol = 'AAPL'
future_predictions = predict_future_prices(stock_symbol)

```
## Result

![image](https://github.com/user-attachments/assets/92437257-5473-4717-8411-4b7d1baf9978)
**WARNING:** This model is just a prediction model. I cannot accept any responsibility.

# Training Code:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a8i6bOKRw9h-gzO4S1GkRa71mZITuMge?usp=sharing)

# Fine-tuning Space:
Using ZeroGPU support and LoRA training with any stock market. (You can find stock symbols on Yahoo Finance)

[Hugging Face Space](https://huggingface.co/spaces/Q-bert/StockLlama-TrainOnAnyStock)

# For more:

You can reach me on,

[Linkedin](https://www.linkedin.com/in/talha-r%C3%BCzgar-akku%C5%9F-1b5457264/)

[Twitter](https://x.com/TalhaRuzga35606)

[Hugging Face](https://huggingface.co/Q-bert)
