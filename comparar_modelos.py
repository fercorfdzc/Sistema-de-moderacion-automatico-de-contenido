import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import os

def generar_comparacion():
    # 1. Configuración de rutas
    # Intentamos detectar la ubicación del dataset
    dataset_path = "data/dataset_ready_nlp.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "dataset_ready_nlp.csv" # Por si está en la raíz
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        return

    # 2. Cargar Datos
    print(f"Cargando dataset desde: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df['labels'] = df['labels'].astype(int)

    # 3. Split de Datos (Mismo que en el Notebook para consistencia)
    RANDOM_SEED = 42
    df_train, df_temp = train_test_split(
        df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['labels']
    )
    _, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=df_temp['labels']
    )

    # 4. Vectorización TF-IDF para modelos clásicos
    print("Vectorizando texto con TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(df_train['text'])
    X_test_tfidf = vectorizer.transform(df_test['text'])
    y_train = df_train['labels']
    y_test = df_test['labels']

    # 5. Definición y entrenamiento de modelos base
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": LinearSVC(max_iter=1000, dual=False)
    }

    results_list = []
    print("Entrenando y evaluando modelos clásicos...")
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        results_list.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, preds),
            "F1-score": f1_score(y_test, preds, average='macro')
        })

    # 6. Añadir el resultado obtenido por el modelo BERT optimizado
    # Usamos los valores exactos obtenidos en la última ejecución exitosa
    results_list.append({
        "Modelo": "BERT (Optimizado Colab)",
        "Accuracy": 0.9287,
        "F1-score": 0.9287
    })

    # 7. Crear DataFrame final y ordenar
    df_final = pd.DataFrame(results_list)
    df_final = df_final.sort_values(by="F1-score", ascending=False).reset_index(drop=True)

    # 8. Mostrar Resultados
    print("\n" + "="*55)
    print("Table 5. Comparación de modelos de clasificación")
    print("="*55)
    print(df_final.to_string(index=False, formatters={
        'Accuracy': '{:,.4f}'.format,
        'F1-score': '{:,.4f}'.format
    }))
    print("="*55)

    # 9. Guardar a CSV
    output_file = "comparacion_modelos_final.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\nResultados exportados exitosamente a: {output_file}")

if __name__ == "__main__":
    generar_comparacion()
