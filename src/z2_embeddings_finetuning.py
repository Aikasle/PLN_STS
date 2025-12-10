import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import util
from scipy.stats import pearsonr
import math

def main():
    # --- Configuración para Opción 2 (Sentence Embeddings Fine-tuning) ---
    # El enunciado dice: "Se toma el modelo y la arquitectura utilizados en Z1".
    # Usamos el modelo ligero de Z1, pero ahora lo vamos a entrenar.
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    batch_size = 16
    epochs = 4
    model_save_path = './output/z2_siamese_finetuned'

    print(f">>> Iniciando Z2 (Opción 2): Fine-Tuning de {model_name} (Siamese Network)")

    # 1. Cargar Dataset
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    # 2. Preparar Datos para Entrenamiento
    # Para Siamese Networks con CosineSimilarityLoss, necesitamos pares y un score (0-1).
    train_examples = []
    for row in dataset['train']:
        # Normalizamos score de 0-5 a 0-1
        train_examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']], 
            label=row['score'] / 5.0
        ))

    # Creamos el DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 3. Preparar Datos de Validación
    # Usamos EmbeddingSimilarityEvaluator, específico para modelos Siameses.
    # Calcula la similitud coseno entre embeddings y la correlación con las etiquetas humanas.
    val_sentences1 = dataset['validation']['sentence1']
    val_sentences2 = dataset['validation']['sentence2']
    val_scores = [s / 5.0 for s in dataset['validation']['score']]

    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences1, val_sentences2, val_scores, name='sts-val'
    )

    # 4. Inicializar Modelo (Sentence Transformer)
    # Cargamos la arquitectura de Z1
    model = SentenceTransformer(model_name)

    # 5. Definir Función de Pérdida
    # El enunciado especifica: "la función de pérdida estará basada en la similitud del coseno".
    train_loss = losses.CosineSimilarityLoss(model)

    # 6. Entrenar (Fine-Tuning)
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)

    print(">>> Entrenando arquitectura Siamese (Sentence Embeddings)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path
    )

    print(f"\n>>> Entrenamiento finalizado. Modelo guardado en {model_save_path}")

    # 7. Evaluación Final en TEST (Manual con Pearson)
    print(">>> Evaluando modelo final...")
    final_model = SentenceTransformer(model_save_path)
    
    test_sentences1 = dataset['test']['sentence1']
    test_sentences2 = dataset['test']['sentence2']
    test_scores = dataset['test']['score'] # Scores originales 0-5

    # A diferencia del Cross-Encoder, aquí generamos embeddings por separado
    embeddings1 = final_model.encode(test_sentences1, convert_to_tensor=True)
    embeddings2 = final_model.encode(test_sentences2, convert_to_tensor=True)

    # Calculamos la similitud coseno entre pares
    cosine_scores = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()

    # Calculamos Pearson
    pearson_score, _ = pearsonr(cosine_scores, test_scores)
    
    print(f"\n>>> Resultado Final en TEST (Pearson): {pearson_score:.4f}")

if __name__ == "__main__":
    main()
