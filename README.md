# SQuAD dataset with Latent Scope

## Prepare

Firt we use prepare.ipynb to download and ingest the SQuAD dataset into Latent Scope.

We can run the (deduplicated) context through latent scope to embed and create scopes.

Once we have the scopes, we can create a LanceDB table for each scope:

```bash
python makelance.py --directory ~/latent-scope-data --dataset squad --scope_id scopes-001 --metric cosine 
```

This will allow us to calculate hit rates for the questions in the SQuAD dataset.

We use hitrate.ipynb to calculate the hit rate, which also allows us to see the results on the scope visualization.