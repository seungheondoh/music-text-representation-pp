for DATASET in mtg_top50tags mtg_genre mtg_instrument mtg_moodtheme gtzan fma mtat kvt emotify
do
    python train_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --eval_dataset $DATASET

    # add dropout
    python train_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --dropout 0.25 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --dropout 0.25 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --dropout 0.25 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --dropout 0.25 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --dropout 0.25 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --dropout 0.25 --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --dropout 0.25 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --dropout 0.25 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --dropout 0.25 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --dropout 0.25 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --dropout 0.25 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --dropout 0.25 --eval_dataset $DATASET

    # add layer norm
    python train_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --is_norm --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --is_norm --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --is_norm --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --is_norm --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --is_norm --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --is_norm --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --is_norm --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --is_norm --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --is_norm --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --is_norm --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --is_norm --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --is_norm --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --dropout 0.25 --is_norm 1 --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 1e-3 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 1e-3 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 1e-3 --is_norm 1 --eval_dataset $DATASET

    python train_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 16 --lr 5e-4 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 64 --lr 5e-4 --is_norm 1 --eval_dataset $DATASET
    python train_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --is_norm 1 --eval_dataset $DATASET
    python eval_probing.py --probe_type mlp --batch-size 512 --lr 5e-4 --is_norm 1 --eval_dataset $DATASET
done
