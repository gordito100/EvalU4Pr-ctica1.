from fastapi import FastAPI
import pickle

app = FastAPI()

# Cargar el modelo desde el archivo .pkl
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    return {"API": "Modelo de predicción de autenticación de billetes"}

@app.get("/predict")
def predict(variance: float, skewness: float, curtosis: float, entropy: float):
    # Hacer la predicción utilizando el modelo cargado
    prediction = model.predict([[variance, skewness, curtosis, entropy]])

    # Definir las clases y sus descripciones
    classes = {
        0: {
            "name": "Billete auténtico",
            "description": "El billete se considera auténtico."
        },
        1: {
            "name": "Billete falsificado",
            "description": "El billete se considera falsificado."
        }
    }

    # Obtener la descripción y otros detalles según la clase predicha
    predicted_class = int(prediction[0])
    class_name = classes[predicted_class]["name"]
    class_description = classes[predicted_class]["description"]

    # •	Creamos un diccionario de respuesta que contiene la predicción
    response = {
        "prediction": predicted_class,
        "class_name": class_name,
        "class_description": class_description,
        "variance": variance,
        "skewness": skewness,
        "curtosis": curtosis,
        "entropy": entropy

        
    }

    return response
