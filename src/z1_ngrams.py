import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

def main():
    print(">>> Z1.1: Representación basada en N-Grams")
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    
    sentences1 = dataset['sentence1']
    sentences2 = dataset['sentence2']
    human_scores = dataset['score'] # Puntuaciones de 0 a 5

    vectorizer = CountVectorizer()
    
    corpus_total = sentences1 + sentences2
    vectorizer.fit(corpus_total)
    
    v1 = vectorizer.transform(sentences1)
    v2 = vectorizer.transform(sentences2)

    similitudes = cosine_similarity(v1, v2).diagonal()

    pearson_corr, _ = pearsonr(similitudes, human_scores)
    
    print(f"Resultados N-Grams:")
    print(f"---------------------")
    print(f"Muestras evaluadas: {len(human_scores)}")
    print(f"Correlación Pearson: {pearson_corr:.4f}")
    print(f"(Rango esperado: [-1, 1]. Cuanto más alto, mejor) [cite: 26]")

if __name__ == "__main__":
    main()
