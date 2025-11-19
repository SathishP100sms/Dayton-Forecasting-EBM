# Dayton-Forecasting-EMB
The EBM model is designed for short-term load forecasting, predicting the next 1–7 days of hourly load.
It uses the most recent 14–15 days of observations to construct lag and rolling features (e.g., 1-hour, 24-hour, and 168-hour lags, 24-hour and 168-hour rolling statistics), combined with calendar variables (hour of day, day of week, month).
While the model is trained on the full historical dataset, at prediction time it only requires a recent window of data to generate accurate short-term forecasts.
