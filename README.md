# Formula 1 Race Outcome Prediction

This project uses machine learning and historical race data to predict whether a Formula 1 driver will finish on the podium.

The model is trained on race results using features such as:

- Starting grid position
- Constructor (team)
- Circuit

The project also includes visualizations exploring driver performance and race trends.

## Dataset

Formula 1 World Championship Dataset  
Source: Kaggle

The dataset includes historical race results from the Formula 1 World Championship.

Files used:

- results.csv
- drivers.csv
- races.csv
- constructors.csv

## Technologies Used

- Python
- Pandas
- scikit-learn
- Random Forest Classifier
- Matplotlib
- Seaborn

## Machine Learning Pipeline

1. Load and merge race datasets
2. Create features for model training
3. Train a Random Forest classifier
4. Evaluate prediction accuracy
5. Visualize race trends

## Visualizations

The project includes several visualizations:

- Distribution of podium finishes
- Starting grid position vs podium finish
- Top drivers by race wins

These plots help explore the relationship between starting position and race performance.

## Model Output

The model predicts whether a driver will finish on the podium (Top 3).

Example output:

Model Accuracy: ~0.82

## Skills Demonstrated

- Data Cleaning
- Feature Engineering
- Machine Learning Classification
- Sports Data Analytics
- Data Visualization

## Future Improvements

- Include lap time data
- Predict race winners
- Build driver performance ranking models
