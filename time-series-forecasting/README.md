# Financial Time-Series Analysis Using RNN Architectures

## Project Overview
This project focuses on developing machine learning models for time series price data with the business goal of predicting near-term fluctuations in price. Three recurrent neural network (RNN) architectures are compared - Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) - to evaluate their performance in financial forecasting scenarios. The analysis examines both prediction accuracy and computational efficiency to determine the optimal model architecture for this task.

## Data
For this project, I used time series data of IBM stock prices over a five-year period (2019-01-01 to 2024-01-01), obtained from the Yahoo Finance package. The data is structured as a Pandas DataFrame with the following attributes:
* `Date` **[as index]** = The date in the format `YYYY-MM-DD`
* `Open` = Opening IBM stock price on the specified date
* `High` = The highest price obtained on the specified date
* `Low` = The lowest price obtained on the specified date
* `Close` = The closing price on the specified date
* `Adj Close` = Closing price adjusted for dividend distributions or stock splits
* `Volume` = The total number of shares traded on the specified date

After initial analysis, I focused on the closing price time series as the primary target for prediction, as variations in the price over time far exceeded the differences between daily opening, closing, high, and low prices.

## Learning Goals
This project demonstrates proficiency with three types of recurrent neural network architectures used for sequential data analysis and forecasting:

* **Vanilla RNNs** - Simple recurrent units that produce a single hidden state passed to the following unit
* **Long Short-Term Memory (LSTM) Models** - Sophisticated units incorporating cell state and gating mechanisms (forget, input, and output gates) to maintain longer-term context
* **Gated Recurrent Units (GRUs)** - A compromise between vanilla RNNs and LSTMs, with simplified gating mechanisms (reset and update gates)

## Methodology
The analysis followed these key steps:

1. **Data Preparation**: The closing price time series was split into training (75%) and testing (25%) segments, and each was separately scaled using MinMaxScaler to maintain the integrity of the time-dependent data
2. **Data Structuring**: The data was formatted into overlapping 100-day windows (inputs) with the subsequent day's price as the target (output)
3. **Model Development**: Three distinct model architectures were implemented:
   - A vanilla RNN model with four layers and dropout regularization
   - An LSTM model with two LSTM layers and two dense layers
   - A GRU model structured similarly to the vanilla RNN model but with GRU layers
4. **Training and Evaluation**: Each model was trained for an appropriate number of epochs (20 for vanilla RNN and GRU, 12 for LSTM) and evaluated based on prediction accuracy (RMSE) and training efficiency

## Key Findings
The analysis yielded several significant insights:

* **LSTM Superiority**: The LSTM model demonstrated the highest accuracy in predicting time series data, even though it was trained for fewer epochs (12) than the other models
* **GRU Underperformance**: Surprisingly, the GRU model performed worse than the vanilla RNN despite its theoretical advantages and identical structure (replacing vanilla RNN layers with GRU layers)
* **Computational Efficiency**: The GRU model required approximately four times the training time of the vanilla RNN model, while the LSTM model required only about 20% more training time than the vanilla RNN
* **Accuracy-Efficiency Tradeoff**: The LSTM model offered the best balance between prediction accuracy and computational efficiency for this financial forecasting task

## Conclusions
The primary conclusion of this exercise is that the LSTM architecture provides the optimal balance of predictive accuracy and training efficiency for financial time series forecasting. The model successfully captures the patterns in IBM stock price movements, though it tends to underpredict the magnitudes of extreme peaks and troughs in the data.

## Future Work
Several avenues for future research and improvement include:

* Further fine-tuning of the LSTM model to improve its prediction of extreme values
* Investigation of the model's predictive capacity beyond single-day forecasts
* Exploration of model robustness to exogenous market-destabilizing effects
* Incorporation of additional economic indicators and outside information to enhance prediction accuracy

## Technical Implementation
The analysis was implemented using Python with the following key libraries:
* TensorFlow/Keras for neural network implementation
* Pandas and NumPy for data manipulation
* Matplotlib and Seaborn for visualization
* Scikit-learn for data preprocessing and evaluation metrics

This project was completed as part of the IBM Course on Deep Learning and Reinforcement Learning, within the IBM Machine Learning Professional Certification program.