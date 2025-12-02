import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Z1.2: Representación basada en Embeddings (Dispositivo: {device.upper()})")

    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    sentences1 = dataset['sentence1']
    sentences2 = dataset['sentence2']
    human_scores = dataset['score']

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name, device=device)

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    similitudes = cosine_scores.diagonal().cpu().numpy()

    pearson_corr, _ = pearsonr(similitudes, human_scores)

    print(f"Resultados Sentence Embeddings ({model_name}):")
    print(f"---------------------")
    print(f"Muestras evaluadas: {len(human_scores)}")
    print(f"Correlación Pearson: {pearson_corr)

if __name__ == "__main__":
    main()
