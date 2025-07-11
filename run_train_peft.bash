#!/bin/bash

source ~/venv-bongard/bin/activate

cd src

# C scan for every model
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=0.0 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=0.1 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=0.2 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=0.4 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=0.8 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=1.6 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# train_prompt_similarity.py  --model=gemma3_4b --dataset=hoi --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=linear --cl_scaling=3.2 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
# C=0.4 for Phi and Gemma, 1.6 for Pixtral


# LoRA
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=hoi       --learnable_embedding_location=lora   --num_epochs=3  --learning_rate=0.0001 --batch_size=25 --train_limit=4000 --val_limit=100 --cl_scaling=1.6 --contrastive_schedule=cosinel --smooth_cl=True
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=hoi       --learnable_embedding_location=lora   --num_epochs=3  --learning_rate=0.0001 --batch_size=25 --train_limit=4000 --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=openworld --learnable_embedding_location=lora   --num_epochs=20 --learning_rate=0.0001 --batch_size=25 --train_limit=500 --val_limit=100 --cl_scaling=1.6 --contrastive_schedule=cosinel --smooth_cl=True --resolution=200
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=openworld --learnable_embedding_location=lora   --num_epochs=20 --learning_rate=0.0001 --batch_size=25 --train_limit=500 --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant --resolution=200

#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=hoi       --learnable_embedding_location=lora   --num_epochs=3  --learning_rate=0.0001 --batch_size=25 --train_limit=4000 --val_limit=100 --cl_scaling=0.4 --contrastive_schedule=cosinel --smooth_cl=True
#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=hoi       --learnable_embedding_location=lora   --num_epochs=3  --learning_rate=0.0001 --batch_size=25 --train_limit=4000 --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=openworld --learnable_embedding_location=lora   --num_epochs=20 --learning_rate=0.0001 --batch_size=25 --train_limit=500 --val_limit=100 --cl_scaling=0.4 --contrastive_schedule=cosinel --smooth_cl=True
#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=openworld --learnable_embedding_location=lora   --num_epochs=20 --learning_rate=0.0001 --batch_size=25 --train_limit=500 --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant

#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=hoi       --learnable_embedding_location=lora   --num_epochs=3 --learning_rate=0.0001 --batch_size=25 --train_limit=4000 --val_limit=100 --cl_scaling=0.4 --contrastive_schedule=cosinel --smooth_cl=True
#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=hoi       --learnable_embedding_location=lora   --num_epochs=3 --learning_rate=0.0001 --batch_size=25 --train_limit=4000 --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=openworld --learnable_embedding_location=lora   --num_epochs=20 --learning_rate=0.0001 --batch_size=25 --train_limit=500  --val_limit=100 --cl_scaling=0.4 --contrastive_schedule=cosinel --smooth_cl=True
#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=openworld --learnable_embedding_location=lora   --num_epochs=20 --learning_rate=0.0001 --batch_size=25 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant

# full tuning
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=openworld --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=cosinel --cl_scaling=1.6 --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=hoi       --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=cosinel --cl_scaling=1.6 --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=openworld --learnable_embedding_location=full   --num_epochs=20 --cl_scaling=0.0 --contrastive_schedule=constant --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=pixtral   --dataset=hoi       --learnable_embedding_location=full   --num_epochs=20 --cl_scaling=0.0 --contrastive_schedule=constant --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25

#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=openworld --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=cosinel --cl_scaling=0.4 --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=hoi       --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=cosinel --cl_scaling=0.4 --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=openworld --learnable_embedding_location=full   --num_epochs=20 --cl_scaling=0.0 --contrastive_schedule=constant --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=phi       --dataset=hoi       --learnable_embedding_location=full   --num_epochs=20 --cl_scaling=0.0 --contrastive_schedule=constant --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25

#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=openworld --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=cosinel --cl_scaling=0.4 --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=hoi       --learnable_embedding_location=full   --num_epochs=20 --contrastive_schedule=cosinel --cl_scaling=0.4 --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=openworld --learnable_embedding_location=full   --num_epochs=20 --cl_scaling=0.0 --contrastive_schedule=constant --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25
#+ python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=hoi       --learnable_embedding_location=full   --num_epochs=20 --cl_scaling=0.0 --contrastive_schedule=constant --learning_rate=0.0001 --batch_size=25 --trainable_prompt_size=100 --train_limit=500 --val_limit=25


# lr 0.01 outperforms lr 0.0001 sometimes by significant margin
# python3 -u train_prompt_similarity.py --model=phi       --dataset=openworld --learnable_embedding_location=postfix --num_epochs=10 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
# python3 -u train_prompt_similarity.py --model=phi       --dataset=hoi       --learnable_embedding_location=postfix --num_epochs=10 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
# python3 -u train_prompt_similarity.py --model=pixtral   --dataset=openworld --learnable_embedding_location=postfix --num_epochs=10 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
# python3 -u train_prompt_similarity.py --model=pixtral   --dataset=hoi       --learnable_embedding_location=postfix --num_epochs=10 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
# python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=openworld --learnable_embedding_location=postfix --num_epochs=10 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant
# python3 -u train_prompt_similarity.py --model=gemma3_4b --dataset=hoi       --learnable_embedding_location=postfix --num_epochs=10 --learning_rate=0.01 --batch_size=2 --trainable_prompt_size=100 --train_limit=500  --val_limit=100 --cl_scaling=0.0 --contrastive_schedule=constant

