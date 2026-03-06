import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# visual style settings 
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams["figure.figsize"] = (10,6)

#loading datasets
results = pd.read_csv("results.csv")
drivers = pd.read_csv("drivers.csv")
races = pd.read_csv("races.csv")
constructors = pd.read_csv("constructors.csv")

#merging datasets
df = results.merge(drivers, on="driverId")
df = df.merge(races, on="raceId")
df = df.merge(constructors, on="constructorId")

# Create driver name
df["driver"] = df["forename"] + " " + df["surname"]

# Convert finishing position to numeric
df["position"] = pd.to_numeric(df["position"], errors="coerce")

# Create target variable (podium finish)
df["podium"] = df["position"].apply(lambda x: 1 if x <= 3 else 0)

# preparing data for machine learning
data = df[["grid", "constructorId", "circuitId", "podium"]].dropna()

X = data[["grid", "constructorId", "circuitId"]]
y = data["podium"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Podium Distribution
plt.figure()

sns.countplot(
    x=df["podium"],
    palette="viridis"
)

plt.title("Distribution of Podium Finishes", fontsize=16, weight="bold")
plt.xlabel("Podium Finish (1 = Yes, 0 = No)")
plt.ylabel("Number of Results")

sns.despine()

plt.show()

# Grid Position vs Podium Finish
plt.figure()

sns.boxplot(
    x=df["podium"],
    y=df["grid"],
    palette="coolwarm"
)

plt.title("Starting Grid Position vs Podium Finish", fontsize=16, weight="bold")
plt.xlabel("Podium Finish")
plt.ylabel("Starting Grid Position")

sns.despine()

plt.show()

#top drivers by wins
wins = df[df["position"] == 1]["driver"].value_counts().head(10)

plt.figure()

sns.barplot(
    x=wins.values,
    y=wins.index,
    palette="magma"
)

plt.title("Top 10 Drivers by Race Wins", fontsize=16, weight="bold")
plt.xlabel("Number of Wins")
plt.ylabel("Driver")

sns.despine()

plt.show()

#figure saving for readme 
plt.figure()

sns.barplot(
    x=wins.values,
    y=wins.index,
    palette="magma"
)

plt.title("Top 10 Drivers by Race Wins", fontsize=16, weight="bold")
plt.xlabel("Number of Wins")
plt.ylabel("Driver")

sns.despine()

plt.savefig("top_drivers.png", dpi=300, bbox_inches="tight")

plt.show()