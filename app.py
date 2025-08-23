import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Streamlit app
st.title("🌸 Iris Flower Classifier")
st.write("Predict iris species based on sepal and petal measurements")

# Sliders for user input
sepal_length = st.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict
prediction = model.predict(input_data)[0]
predicted_species = iris.target_names[prediction]

# Show result
st.success(f"🌼 Predicted Species: **{predicted_species.capitalize()}**")

# Show image for predicted flower
if predicted_species == "setosa":
    st.image("images/setosa.jpg", caption="Iris Setosa", width=300)
elif predicted_species == "versicolor":
    st.image("images/versicolor.jpg", caption="Iris Versicolor", width=300)
else:
    st.image("images/virginica.jpg", caption="Iris Virginica", width=300)



