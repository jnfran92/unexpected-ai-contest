# MercadoLibre AI contest

Running on TheBebop Server.

## Notes

To Load data in parallel: `train.csv`, chuck it:
    
    mkdir train_chunks
    split -l 500000 train.csv ./train_chunks/train_chunk_ --additional-suffix=.csv

On Mac:
    
     mkdir train_chunks
     split -l 500000 train.csv ./train_chunks/train_chunk_
     cd train_chunks
     for i in *; do mv "$i" "$i.csv"; done


# Requirements

- Tensorflow
- Dask