for DATASET in mtg gtzan fma mtat kvt emotify; do
    for MODEL in tag tag_caption meta_tag_caption meta_tag_caption_sim; do
        if [ $DATASET == "mtg" ]; then
            python extractor.py --gpu $1 --eval_dataset $DATASET --model_type $MODEL --num_chunks 12 # 2min
        else
            python extractor.py --gpu $1 --eval_dataset $DATASET --model_type $MODEL --num_chunks 3 # 30sec
        fi
    done
done