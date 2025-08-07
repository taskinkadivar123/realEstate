# Dragon Real Estate Investment Analysis

This project analyzes real estate investment opportunities using machine learning. It is based on a dataset containing housing-related features like the number of rooms, crime rate, pollution levels, etc., and predicts the **median value of owner-occupied homes (MEDV)**.

This project is inspired by "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" and guided by [CodeWithHarry](https://www.youtube.com/c/CodeWithHarry).

## Objective
Dragon Real Estate wants to automate price prediction to identify undervalued properties based on historical data. Previously, manual expert estimation had a 25% error rate. This project builds a supervised machine learning regression model to improve that accuracy.

## 🧠 Key Concepts
- **Supervised Learning**: Predict house price (a continuous value)
- **Regression**: Predict numerical value (not categories)
- **Performance Metric**: RMSE (Root Mean Squared Error)
- **Data Split Strategy**: Stratified Shuffle Split (preserves class proportions)
- **Feature Engineering**: Derived feature `TAXRM = TAX / RM`

## Dataset Features Explained

| Feature   | Description |
|-----------|-------------|
| CRIM      | Crime rate |
| ZN        | Proportion of land for large homes |
| INDUS     | % of non-retail business acres |
| CHAS      | Bounded by Charles River (1=yes) |
| NOX       | Air pollution |
| RM        | Avg number of rooms per dwelling |
| AGE       | % of houses built before 1940 |
| DIS       | Distance to job centers |
| RAD       | Highway access index |
| TAX       | Property tax |
| PTRATIO   | Pupil-teacher ratio |
| B         | Proportion of Black residents |
| LSTAT     | % of low-income residents |
| MEDV      | Median value of owner-occupied homes |


**This same dataset more clear i understand in the project which is natural way understand so there check it(ML_project.py)**

## 🧪 Workflow
1. **Load Data** from `data.csv`
2. **Explore & Visualize** using histograms and scatter plots
3. **Split Data** using `train_test_split` and `StratifiedShuffleSplit`
4. **Check Correlations** to understand feature importance
5. **Feature Engineering** to create new meaningful variables (e.g., `TAXRM`)
6. **Plot Data** to detect outliers and data patterns
7. **Train ML Model** (coming in next steps...)

## 📈 Visual Insights
- **MEDV vs RM** → Positive correlation: More rooms → higher price
- **MEDV vs LSTAT** → Negative correlation: More low-income residents → lower price
- **MEDV vs ZN** → Unclear correlation, may depend on area
- **MEDV vs TAXRM** → Helps in detecting poor/high tax vs room ratio areas

## 🛠️ Requirements
- Python 3.x
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`

Install using:
pip install pandas matplotlib numpy scikit-learn
