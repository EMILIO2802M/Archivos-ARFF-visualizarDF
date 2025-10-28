from django.shortcuts import render
import pandas as pd
import arff
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('arff_file'):
        arff_file = request.FILES['arff_file']

        if not arff_file.name.endswith('.arff'):
            context['error_message'] = 'Error: Please upload a valid .arff file.'
            return render(request, 'index.html', context)

        try:
            # Leemos el archivo que se subió
            file_content = arff_file.read().decode('utf-8')
            file_stream = io.StringIO(file_content)

            # Cargamos los datos del archivo ARFF
            dataset = arff.load(file_stream)

            # Sacamos los nombres de las columnas de los atributos
            columns = [attr[0] for attr in dataset['attributes']]

            # Creamos el DataFrame con los datos
            df = pd.DataFrame(dataset['data'], columns=columns)

            # --- AQUI EMPIEZA LO BUENO DE LA PRECISION ---
            
            # Separamos las caracteristicas y la variable objetivo
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            # Vemos cuales son categoricas y cuales numericas
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            numerical_features = X.select_dtypes(include=['number']).columns

            # Hacemos un transformador de columnas para preprocesar los datos
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])

            # Codificamos la variable objetivo
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Partimos los datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            
            # Preprocesamos los datos
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Creamos y entrenamos el arbolito de decision
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train_processed, y_train)

            # Hacemos las predicciones
            y_pred = clf.predict(X_test_processed)

            # Calculamos la precision
            accuracy = accuracy_score(y_test, y_pred)
            context['accuracy'] = f'{accuracy:.2%}'

            # --- AQUI TERMINA LO BUENO DE LA PRECISION ---

            # Pasamos el dataframe a HTML para que se vea bonito
            context['dataframe_html'] = df.to_html(classes='table table-striped table-hover', index=False, justify='left')

        except Exception as e:
            context['error_message'] = f'Error processing file: {e}'

    return render(request, 'index.html', context)