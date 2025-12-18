import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr

MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v2'

def evaluar_idioma(model, dataset_name, subset, lang_label):
    """
    Helper para evaluar el modelo en un idioma específico.
    """
    print(f"  ...Evaluando en {lang_label} ({subset})...")
    
    if subset == 'default':
        ds = load_dataset(dataset_name, split='test')
    else:
        ds = load_dataset(dataset_name, subset, split='test')
        
    s1 = ds['sentence1']
    s2 = ds['sentence2']
    try:
        scores = ds['similarity_score']
    except: 
        scores = ds['score']

    emb1 = model.encode(s1, convert_to_tensor=True)
    emb2 = model.encode(s2, convert_to_tensor=True)
    
    cosine_sims = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
    
    pearson, _ = pearsonr(cosine_sims, scores)
    return pearson

def main():
    print(f">>> Iniciando Evaluación Multilingüe (Zero-Shot) con {MODEL_NAME}")
    
    model = SentenceTransformer(MODEL_NAME)
    
    print("\n[FASE ÚNICA] Evaluación Base sin Fine-Tuning")
    
    p_en = evaluar_idioma(model, "mteb/stsbenchmark-sts", "default", "INGLÉS")
    p_es = evaluar_idioma(model, "mteb/stsb_multi_mt", "es", "ESPAÑOL")
    p_zh = evaluar_idioma(model, "mteb/stsb_multi_mt", "zh", "CHINO")
    
    print("-" * 30)
    print(f"Resultados Zero-Shot (Pearson):")
    print("-" * 30)
    print(f"  - Inglés:  {p_en:.4f}")
    print(f"  - Español: {p_es:.4f}")
    print(f"  - Chino:   {p_zh:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
