# for DATASET in gtzan fma mtat kvt emotify mtg_top50tags mtg_genre mtg_instrument mtg_moodtheme
for DATASET in mtg_top50tags mtg_genre mtg_instrument mtg_moodtheme
do
    python eval_zs.py --eval_dataset $DATASET
done
