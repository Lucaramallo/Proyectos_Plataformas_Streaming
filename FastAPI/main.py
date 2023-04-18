from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI(title = 'Prractica Individual 1',
              description= 'Primer proyecto individual'
              )



# Leer el archivo CSV y crear un DataFrame
df_merged = pd.read_csv('../ETL/Data/merged.csv')
# Leer el archivo pickle y guardar los datos en un DataFrame
merged_df = pd.read_pickle('../ETL/Data/merged.pickle')



@app.get("/")
def read_root():
    return {"Bienvenido a la entrega de ": "Práctica individual 1 - Luca Ramallo FT-09"}




# output provistos:


@app.get('/get_max_duration/{anio}/{plataforma}/{dtype}')
def get_max_duration(year, platform, duration_type):

    # Filtrar el DataFrame para incluir solo las películas
    movies_df = df_merged[df_merged['type'] == 'movie']
    
    # Convertir la columna 'duration_int' a valores numéricos
    movies_df['duration_int'] = pd.to_numeric(movies_df['duration_int'], errors='coerce')
    
    # Filtrar por año, plataforma y tipo de duración
    filtered_df = movies_df[(movies_df['release_year'] == year) & 
                            (movies_df['plataforma'] == platform) &
                            (movies_df['duration_type'] == duration_type)]
    
    # Devolver el título de la película con la duración máxima
    max_duration_movie = filtered_df.loc[filtered_df['duration_int'].idxmax(), 'title']
    
    # Crear el diccionario de salida
    output_dict = {
        'plataforma': platform,
        'tipo_duracion': duration_type,
        'año': year,
        'titulo': max_duration_movie
    }
    
    return output_dict



@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(plataforma: str, scored: float, anio: int):
        # Filtrar sólo las películas del año y plataforma solicitados
    movies = df_merged[(df_merged['plataforma'] == plataforma) & 
                       (df_merged['type'] == 'movie') &
                       (df_merged['release_year'] == anio)]
    
    # Contar las películas que cumplen con el puntaje mínimo
    count = (movies['rating_movie_user'] >= scored).sum()
    
    return {
        'plataforma': plataforma,
        'cantidad': count,
        'anio': anio,
        'score': scored
    }


@app.get('/get_count_platform/{plataforma}')
def get_count_platform(plataforma: str):
    platform_df = df_merged[df_merged['type'] == 'movie']
    respuesta = len(platform_df[platform_df['plataforma'] == plataforma])
    return {'plataforma': plataforma, 'peliculas': respuesta}



@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(plataforma: str, anio: int):
    # get data for the given platform and year
    platform_data = df_merged[df_merged['plataforma'] == plataforma]
    year_data = platform_data[platform_data['release_year'] == anio]

    # concatenate the cast and director fields
    year_data['combined'] = year_data['cast'] + ',' + year_data['director']

    # get the actor or director that appears most frequently
    combined_counts = year_data['combined'].value_counts()
    max_count = combined_counts.max()
    top_actors = combined_counts[combined_counts == max_count]
    most_frequent_actor = top_actors.index[0]

    # separate the actor and director names
    most_frequent_actor_parts = most_frequent_actor.split(',')
    most_frequent_cast = most_frequent_actor_parts[0]
    most_frequent_director = most_frequent_actor_parts[1]

    # create the output dictionary
    return{
        'plataforma': plataforma,
        'anio': anio,
        'actor': most_frequent_cast,
        'apariciones': max_count}


@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    # Filtrar el DataFrame por el rating de audiencia dado
    contents = df_merged[df_merged['rating'] == rating]
    # Obtener el número total de contenidos
    num_contents = contents.shape[0]
    # Devolver una respuesta en formato JSON
    respuesta = {'rating': rating, 'num_contents': num_contents}
    return {'rating': rating, 'contenido': num_contents}


@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_county(tipo: str, pais: str, anio: int):
    # filtrar los datos según el tipo de contenido, país y año
    df_filt = df_merged[(df_merged['type'] == tipo) & (df_merged['country'] == pais) & (df_merged['release_year'] == anio)]
    # contar el número de filas
    respuesta = len(df_filt)
    # crear un diccionario con los resultados y devolverlo
    result = {'pais': pais, 'anio': anio, 'peliculas': tipo, 'count': respuesta}
    return {'pais': pais, 'anio': anio, 'peliculas': respuesta}



@app.get('/get_recomendation/{title}')
def get_recomendation(title,):
        # Obtener la película dada por su título
    pelicula = merged_df[merged_df['title'] == titulo].iloc[0]
    
    # Crear una lista con los títulos de todas las películas
    titulos = merged_df['title'].tolist()
    
    # Crear una matriz de recuento de términos de la columna "listed_in"
    vectorizer = CountVectorizer()
    generos_matriz = vectorizer.fit_transform(merged_df.iloc[:, 7:])
    
    # Obtener el índice de la película dada
    idx = titulos.index(titulo)
    
    # Obtener los vectores de características de todas las películas
    pelicula_features = generos_matriz[idx]
    features_matriz = generos_matriz
    
    # Calcular la similitud del coseno entre la película dada y todas las demás películas
    similarities = cosine_similarity(pelicula_features, features_matriz).flatten()
    
    # Obtener los índices de las películas más similares
    similar_indices = similarities.argsort()[-2:-1][::-1]
    
    # Obtener los títulos de las películas más similares
    similar_titulos = [titulos[i] for i in similar_indices]
    
    return {'recomendacion':similar_titulos}