# PLN_STS

El objetivo de este proyecto es diseñar y evaluar modelos que midan la similitud de significado entre dos oraciones

Usaremos [STS-Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) hosteado en HuggingFace como dataset.

## Z0 

En el z0 analizaremos un poco el dataset, para entender un poco mejor con que datos estamos trabajando. 

Estos son los datos

| Frase 1 | Frase 2 | Puntuacion Humana |
|---------|---------|-------------------|
| A girl is styling her hair. | A girl is brushing her hair. | 2.5 | 
| A group of men play soccer on the beach. | A group of boys are playing soccer on the beach. | 3.6 | 
| One woman is measuring another woman's ankle. | A woman measures another woman's ankle. | 5.0 | 
| A man is cutting up a cucumber. | A man is slicing a cucumber. | 4.2 | 
| A man is playing a harp. | A man is playing a keyboard. | 1.5 | 

El dataset original tiene mas informacion, pero nosotros nos vamos a centrar
en las columnas "sentence1", "sentence2" y "score".
Dos frases y la puntuacion dada por un humano sobre su similitud.


## Z1

En el z1 trabajaremos la similitud sémantica no supervisada. 
Lo vamos a hacer de dos maneras:
 * [n-grams](https://en.wikipedia.org/wiki/Bag-of-words_model) (mas especificamente "Bag of Words")
 * [sentence embeddings](https://en.wikipedia.org/wiki/Sentence_embedding)


Para medir cuanto se parecen los resultados obtenidos con 
estos metodos, y los valores del dataset he usado la 
[Similitud de Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
Especificamente la implementacion de [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

Esta metrica mide cuanta correlacion hay entre dos conjuntos. En nuestro caso entre los resultados de los "modelos" 
y los origininales que han sido anotados por personas. Devuelve un numero entre [-1, 1] que representa la correlacion. 

Como en Z1 no hay que entrenar nada, usaremos el split "test" del dataset, para no cargar tantos datos.

### Z1.1 n-grams

Para hacer el "Bag of words" usamos _CountVectorizer_. Con esto creamos las representaciones vectoriales de cada frase
y luego medimos las differencias entre las frases con la [formula del coseno](https://en.wikipedia.org/wiki/Cosine_similarity)


Despues de ejecutar [z1_ngrams](./src/z1_ngrams.py) obtenemos los siguientes resultados:
```
Muestras evaluadas: 1379
Correlación Pearson: 0.5705
```

La correlacion de Pearson es bastante pobre, pero para ser el modelo mas simple 
que usaremos no esta mal. 

### Z1.2 sentence embeddings

Para hacer el sentence embeddings usaremos un model ya entrenado con millones de frases 
(sentence-transformers/all-MiniLM-L6-v2), que 
representara nuestra frase con un vector, de una manera mucho mas efficiente que el Bag of Words.

Calculamos la similitud de pearson igual que en el anterior

Despues de ejecutar [z1_embeddings.py](./src/z1_embeddings.py) obtenemos los siguientes resultados:

```
Muestras evaluadas: 1379
Correlación Pearson: 0.8274
```

La correlacion pearson es bastante mejor. El modelo usado es
bastante mejor. Pero aun podemos mejorar estos resultados, entrenando
el modelo con nuestros propios datos.

## Z2

Para este apartado vamos a usar otro metodo, el fine tunning. En vez de entrenar un modelo desde cero o cargar un modelo generico, he cargado un modelo estandar y lo he entrenado con mis datos. Con esto he creado un modelo mas ajustado a mis necesidades, sin tener que entrenarlo desde cero.

En este caso voy a ajustar un model de _sentence embeddings_ especificamente _sentence-transformers/all-MiniLM-L6-v2_ (el cual he usado en Z1.2

Simplemente cargamos los datos y los estructuramos en un _DataLoader_ (train y val). Luego cargamos el modelo, definimos la funcion de perdidia (CosineSimilarityLoss, al igual que en Z1.2) y lo entrenamos con el _DataLoader_ de entrenmaiento. 

Una vez entrenado miramos los resultados con el conjunto de validacion y los compramos con los originales con la correlacion de pearson.

Estos han sido los resultados:
```
Correlación Pearson: 0.8595
``` 

Podemos ver que es mejor que la del _sentence embeddings_ que no han sido ajustados. 
