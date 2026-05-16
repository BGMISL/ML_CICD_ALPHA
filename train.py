import pandas as pd
import joblib
from app.model import train_model

data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [20, 40, 60, 80, 100]
}
df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

# Train the model
model = train_model(X, y)
# Save the model
joblib.dump(model, 'models/model.pkl')
print("Model trained and saved successfully.")