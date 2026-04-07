import argparse
import sys
from utils import getDataset, setup_synapse_dataset, preprocess_synapse

def main():
    parser = argparse.ArgumentParser(description="Setup del dataset Synapse")
    parser.add_argument("--random_seed", type=int, default=None, help="Seed per la divisione casuale del dataset")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Percentuale di dati per il training (default: 0.6)")
    args = parser.parse_args()

    print("STEP 1/3: Download del dataset da Synapse")
    result = getDataset()
    if result is None:
        print(" Download fallito. Interruzione.")
        sys.exit(1)

    print("STEP 2/3: Estrazione e organizzazione del dataset")
    result = setup_synapse_dataset()
    if not result:
        print(" Setup fallito. Interruzione.")
        sys.exit(1)

    print("STEP 3/3: Preprocessing del dataset")
    preprocess_synapse(random_seed=args.random_seed, train_ratio=args.train_ratio)

if __name__ == "__main__":
    main()