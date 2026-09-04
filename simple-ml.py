import numpy as np
from sklearn.linear_model import LinearRegression

# 1.
hours_studied = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

test_scores = np.array([50, 60, 70, 80, 90])

# 2.
model = LinearRegression()

# 3.
model.fit(hours_studied, test_scores)

# 4.
new_student_hours = np.array([4]).reshape(-1, 1)
predicted_score = model.predict(new_student_hours)

# 5.
print(f"Prediction: If a student studies for 4 hours, they will score {predicted_score[0]:.0f}")

print("\nHow it figured this out:")
print(f"The model calculated the base score (intercept) is {model.intercept_:.0f}")
print(f"The model calculated each hour adds (coefficient) {model.coef_[0]:.0f} points")
print(f"The formula it learned on its own: Score = ({model.coef_[0]:.0f} * Hours) + {model.intercept_:.0f}")