import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import math
import os

def evaluar_con_pearson(nombre_exp, preds, reales):
    score, _ = pearsonr(preds, reales)
    print(f"  -> {nombre_exp}: Pearson = {score:.4f}")
    return score

def main():
    print(">>> Iniciando Z3: Transferencia Cross-Lingüe (Español)")
    
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import math
import os

def evaluar_con_pearson(nombre_exp, preds, reales):
    score, _ = pearsonr(preds, reales)
    print(f"  -> {nombre_exp}: Pearson = {score:.4f}")
    return score

def main():
    print(">>> Iniciando Z3: Transferencia Cross-Lingüe (Español)")
    
    print(">>> Cargando dataset STS en Español (stsb_multi_mt)...")
    dataset = load_dataset("mteb/stsb_multi_mt", "es")
    
    test_s1 = dataset['test']['sentence1']
    test_s2 = dataset['test']['sentence2']
    test_scores = dataset['test']['similarity_score'] # 0-5

    print("\n[Experimento 1] N-Grams en Español")
    vectorizer = CountVectorizer()
    vectorizer.fit(test_s1 + test_s2)
    v1 = vectorizer.transform(test_s1)
    v2 = vectorizer.transform(test_s2)
    sims_ngram = cosine_similarity(v1, v2).diagonal()
    evaluar_con_pearson("N-Grams (Español)", sims_ngram, test_scores)

    print("\n[Experimento 2] Modelo Z2 (Entrenado en Inglés) evaluado en Español")
    path_z2 = 'sentence-transformers/all-MiniLM-L6-v2'
    # path_z2 = './output/z2_siamese_finetuned'
    
    # if os.path.exists(path_z2):
    if True:
        model_en = SentenceTransformer(path_z2)
        emb1 = model_en.encode(test_s1, convert_to_tensor=True)
        emb2 = model_en.encode(test_s2, convert_to_tensor=True)
        sims_en = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
        evaluar_con_pearson("Z2 English Model", sims_en, test_scores)
    else:
        print("  [!] No se encontró el modelo Z2. Ejecuta primero z2_bert_finetuning.py")

    print("\n[Experimento 3] Entrenando Modelo Multilingüe con datos en Español...")
    
    model_multi_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    model_multi = SentenceTransformer(model_multi_name)
    
    train_examples = []
    for row in dataset['train']:
        train_examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=row['similarity_score'] / 5.0 # Normalizar
        ))
    
    batch_size = 16
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model_multi)
    
    model_multi.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1, 
        warmup_steps=100,
        output_path='./output/z3_multilingual_es'
    )
    
    print(">>> Evaluando modelo Multilingüe en Español...")
    emb1_m = model_multi.encode(test_s1, convert_to_tensor=True)
    emb2_m = model_multi.encode(test_s2, convert_to_tensor=True)
    sims_multi = util.cos_sim(emb1_m, emb2_m).diagonal().cpu().numpy()
    
    evaluar_con_pearson("Modelo Multilingüe (Fine-tuned ES)", sims_multi, test_scores)

if __name__ == "__main__":
    main()  
    print(">>> Cargando dataset STS en Español (stsb_multi_mt)...")
    dataset = load_dataset("mteb/stsb_multi_mt", "es")
    
    test_s1 = dataset['test']['sentence1']
    test_s2 = dataset['test']['sentence2']
    test_scores = dataset['test']['similarity_score'] # 0-5

    print("\n[Experimento 1] N-Grams en Español")
    vectorizer = CountVectorizer()
    vectorizer.fit(test_s1 + test_s2)
    v1 = vectorizer.transform(test_s1)
    v2 = vectorizer.transform(test_s2)
    sims_ngram = cosine_similarity(v1, v2).diagonal()
    evaluar_con_pearson("N-Grams (Español)", sims_ngram, test_scores)

    print("\n[Experimento 2] Modelo Z2 (Entrenado en Inglés) evaluado en Español")
    path_z2 = './output/z2_siamese_finetuned'
    
    if os.path.exists(path_z2):
        model_en = SentenceTransformer(path_z2)
        emb1 = model_en.encode(test_s1, convert_to_tensor=True)
        emb2 = model_en.encode(test_s2, convert_to_tensor=True)
        sims_en = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
        evaluar_con_pearson("Z2 English Model", sims_en, test_scores)
    else:
        print("  [!] No se encontró el modelo Z2. Ejecuta primero z2_bert_finetuning.py")

    print("\n[Experimento 3] Entrenando Modelo Multilingüe con datos en Español...")
    
    model_multi_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    model_multi = SentenceTransformer(model_multi_name)
    
    train_examples = []
    for row in dataset['train']:
        train_examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']],
            label=row['similarity_score'] / 5.0 # Normalizar
        ))
    
    batch_size = 16
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model_multi)
    
    model_multi.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1, 
        warmup_steps=100,
        output_path='./output/z3_multilingual_es'
    )
    
    print(">>> Evaluando modelo Multilingüe en Español...")
    emb1_m = model_multi.encode(test_s1, convert_to_tensor=True)
    emb2_m = model_multi.encode(test_s2, convert_to_tensor=True)
    sims_multi = util.cos_sim(emb1_m, emb2_m).diagonal().cpu().numpy()
    
    evaluar_con_pearson("Modelo Multilingüe (Fine-tuned ES)", sims_multi, test_scores)

if __name__ == "__main__":
    main()
