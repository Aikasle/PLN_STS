from datasets import load_dataset

def main():
    print(f">>> Z0: Informacion sobre el dataset")
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    sentences1 = dataset['sentence1']
    sentences2 = dataset['sentence2']
    human_scores = dataset['score']

    print(sentences1[:10]) 
    print(sentences2[:10]) 
    print(human_scores[:10]) 
    print("Frase 1 | Frase 2 | Puntuacion Humana")
    for sen1, sen2, hs in zip(sentences1[:5], sentences2[:5], human_scores[:5]):
        print(f"{sen1} | {sen2} | {hs}")


if __name__ == "__main__":
    main()
