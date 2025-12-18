import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from scipy.stats import pearsonr
import math

def main():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'  
    batch_size = 16
    epochs = 4
    model_save_path = './output/z4_finetuned_zh'

    print(f">>> Iniciando Z4: Fine-Tuning Directo en CHINO sobre {model_name}")

    print(">>> Cargando dataset STS en Chino (stsb_multi_mt)...")
    dataset = load_dataset("mteb/stsb_multi_mt", "zh")
    
    print("\n--- Evaluación Inicial (Antes de entrenar) ---")
    model = SentenceTransformer(model_name)
    
    test_s1 = dataset['test']['sentence1']
    test_s2 = dataset['test']['sentence2']
    test_scores = dataset['test']['similarity_score']

    emb1 = model.encode(test_s1, convert_to_tensor=True)
    emb2 = model.encode(test_s2, convert_to_tensor=True)
    sims_inicial = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
    
    pearson_inicial, _ = pearsonr(sims_inicial, test_scores)
    print(f"Pearson Inicial (Zero-Shot): {pearson_inicial:.4f}")

    print("\n--- Preparando datos para Fine-Tuning ---")
    train_examples = []
    for row in dataset['train']:
        train_examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']], 
            label=row['similarity_score'] / 5.0  # Normalizamos score 0-1
        ))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    val_s1 = dataset['test']['sentence1']
    val_s2 = dataset['test']['sentence2']
    val_scores = [s / 5.0 for s in dataset['test']['similarity_score']]
    
    evaluator = EmbeddingSimilarityEvaluator(
        val_s1, val_s2, val_scores, name='sts-val-zh'
    )

    train_loss = losses.CosineSimilarityLoss(model)

    print(f">>> Entrenando el modelo {model_name} con datos en CHINO...")
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path
    )
    
    print(f"\n>>> Entrenamiento finalizado. Modelo guardado en {model_save_path}")

    print("\n--- Evaluación Final (Después de entrenar) ---")
    final_model = SentenceTransformer(model_save_path)
    
    emb1_f = final_model.encode(test_s1, convert_to_tensor=True)
    emb2_f = final_model.encode(test_s2, convert_to_tensor=True)
    
    sims_final = util.cos_sim(emb1_f, emb2_f).diagonal().cpu().numpy()
    pearson_final, _ = pearsonr(sims_final, test_scores)
    
    print(f"Pearson Final (Fine-tuned ZH): {pearson_final:.4f}")
    print(f"Mejora obtenida: {pearson_final - pearson_inicial:.4f}")

if __name__ == "__main__":
    main()
