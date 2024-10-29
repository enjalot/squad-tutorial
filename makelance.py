# convert a scope to a lancedb database
# Example usage:
# python makelance.py --directory ~/latent-scope-data --dataset squad --scope_id scopes-001 --metric cosine 

# creates a table with the same name as the scope id in the lancedb folder of the dataset


import argparse
import os
import json
import h5py
import lancedb
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Convert a scope to a LanceDB database")
    parser.add_argument("--directory", help="Directory containing the scope", type=str, default="~/latent-scope-data")
    parser.add_argument("--dataset", help="Name of the dataset", type=str)
    parser.add_argument("--scope_id", help="ID of the scope to convert", type=str)
    parser.add_argument("--metric", help="Metric to use for the index", type=str, default="cosine")
    args = parser.parse_args()

    dataset_path = os.path.join(args.directory, args.dataset)
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    # Load the scope
    scope_path = os.path.join(dataset_path, "scopes")

    print(f"Loading scope from {scope_path}")
    scope_df = pd.read_parquet(os.path.join(scope_path, f"{args.scope_id}-input.parquet"))
    scopes_meta = json.load(open(os.path.join(scope_path, f"{args.scope_id}.json")))

    print(f"Loading embeddings from {dataset_path}/embeddings/{scopes_meta['embedding_id']}.h5")
    embeddings = h5py.File(os.path.join(dataset_path, "embeddings", f"{scopes_meta['embedding_id']}.h5"), "r")

    db_uri = os.path.join(dataset_path, "lancedb")
    db = lancedb.connect(db_uri)

    print(f"Converting embeddings to numpy arrays")
    scope_df["vector"] = [np.array(row) for row in embeddings['embeddings']]

    table_name = args.scope_id + "_" + args.metric

    # Check if the table already exists
    if args.scope_id in db.table_names():
        # Remove the existing table and its index
        db.drop_table(table_name)
        print(f"Existing table '{table_name}' has been removed.")

    print(f"Creating table '{table_name}'")
    tbl = db.create_table(table_name, scope_df)

    print(f"Creating index on table '{table_name}'")
    vector_dim = embeddings['embeddings'].shape[1]
    num_sub = min(96, max(1, vector_dim // 2))
    tbl.create_index(num_partitions=256, num_sub_vectors=num_sub, metric=args.metric)

    model_name = scopes_meta['embedding']['model_id'][2:].replace("___", "/")
    # Prepare metadata
    metadata = {
        "directory": args.directory,
        "scope_id": args.scope_id,
        "dataset": args.dataset,
        "metric": args.metric,
        "db_uri": db_uri,
        "table_name": table_name,
        "embedding_id": scopes_meta['embedding_id'],
        "model_name": model_name,
    }

    # Save metadata as JSON
    if not os.path.exists("scopes"):
        os.makedirs("scopes")

    metadata_path = os.path.join("scopes", f"{table_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
