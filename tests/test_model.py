import pandas as pd
from app.model import train_model

def test_train_model():
    # Sample data
    data = {
        'Hours': [1, 2, 3, 4, 5],
        'Marks': [20, 40, 60, 80, 90]
    }
    df = pd.DataFrame(data)
    
    X = df[['Hours']]
    y = df['Marks']
    
    model = train_model(X, y)
    

    prediction = model.predict([[6]])
    
    assert prediction[0] >0