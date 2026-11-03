!pip install -U gdown

!gdown "https://drive.google.com/file/d/1ciD6wLZH-qGeFtyANXlQe_jaiCSxZIN8/view?usp=sharing"
!gdown "https://drive.google.com/file/d/1kfT2SVPFux8jZ-0XWDCuu_S1mBaRbx1z/view?usp=sharing"
!gdown "https://drive.google.com/file/d/1AqAQ5h6vkOIkNfdKNwrHPUA8fkaqYCGO/view?usp=sharing"
!gdown "https://drive.google.com/file/d/1NyHMzVoVCtzN7E38s4fdKgrjCFYUy722/view?usp=sharing"

# !gdown --folder \
#   "https://drive.google.com/drive/folders/1bWm54T-dFAsU4aWPsFQ1eVk1Fi6va1ES" \
#   -O "/content/KidnapRAG Datasets/ReAct/datasets/hotpotqa" \
#   --continue \
#   --retries 5

!git clone https://github.com/soroushhashemifar/KidnapRAG.git

!mkdir -p "/content/KidnapRAG/ReAct/datasets/hotpotqa/e5_index"
!mv "/content/embedding-shard-0.pt" "/content/KidnapRAG/ReAct/datasets/hotpotqa/e5_index"
!mv "/content/corpus.jsonl" "/content/KidnapRAG/ReAct/datasets/hotpotqa"
!unzip /content/Chain_Link_and_Mal_Ins_DOCS-20261102T093254Z-1-001.zip
!mv /content/Chain_Link_and_Mal_Ins_DOCS /content/KidnapRAG/ReAct/datasets

!mkdir -p "/content/KidnapRAG/Webthinker/datasets/hotpotqa_corpus/e5_index_hotpotqa_corpus"
!mv "/content/_embedding-shard-0.pt" "/content/KidnapRAG/Webthinker/datasets/hotpotqa_corpus/e5_index_hotpotqa_corpus/embedding-shard-0.pt"
!mv "/content/hotpotqa_corpus.jsonl" "/content/KidnapRAG/Webthinker/datasets/hotpotqa_corpus"
!unzip /content/hotpotqa-20261102T103131Z-1-001.zip
!mv /content/hotpotqa /content/KidnapRAG/Webthinker/datasets
!unzip /content/hotpotqa_poisoned_corpus-20261102T103944Z-1-001.zip
!mv /content/hotpotqa_poisoned_corpus /content/KidnapRAG/Webthinker/datasets
!unzip /content/Chain_Link_and_Mal_Ins_DOCS-20261102T112904Z-1-001.zip
!mv /content/Chain_Link_and_Mal_Ins_DOCS/ "/content/KidnapRAG/Webthinker/datasets"

# !mv "/content/KidnapRAG Datasets/ReAct/datasets" /content/KidnapRAG/ReAct
# !mv "/content/KidnapRAG Datasets/Webthinker/datasets" /content/KidnapRAG/Webthinker

!pip install -q condacolab
import condacolab
condacolab.install()

%cd /content/KidnapRAG/
!sudo apt install python3.10
!conda create -n KidnapRAG python=3.10 -y
!conda activate KidnapRAG
!pip install -r requirements.txt

# ReAct - hotpotqa

%cd /content/KidnapRAG/ReAct/ReAct

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method clean \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python generate_attack_doc_kill.py \
    --input_path /KidnapRAG/ReAct/results/adv_targeted_results/hotpotqa_seed1_clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type ours \
    --max_samples 10

%cd /content/KidnapRAG/ReAct
!python concat.py --dataset hotpotqa --gen_model qwen7b --mode ours

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BM_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BM_qwen7b \
    --model_name intfloat/e5-large-v2

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCM_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCM_qwen7b \
    --model_name intfloat/e5-large-v2

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCM_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCM_qwen7b \
    --model_name intfloat/e5-large-v2

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCCM_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCCM_qwen7b \
    --model_name intfloat/e5-large-v2

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCCCM_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_BCCCCM_qwen7b \
    --model_name intfloat/e5-large-v2

%cd /content/KidnapRAG/ReAct/ReAct
!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method BM \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method BCM \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method BCCM \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method BCCCM \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method BCCCCM \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

## w/o B, M, Chaining (Table 2)

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python generate_attack_doc_kill.py \
    --input_path /KidnapRAG/ReAct/results/adv_targeted_results/hotpotqa_seed1_clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type wo_B \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python generate_attack_doc_kill.py \
    --input_path /KidnapRAG/ReAct/results/adv_targeted_results/hotpotqa_seed1_clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type ours_wo_sr \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python generate_attack_doc_kill.py \
    --input_path /KidnapRAG/ReAct/results/adv_targeted_results/hotpotqa_seed1_clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type ours_wo_Chaining \
    --max_samples 10

%cd /content/KidnapRAG/ReAct
!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_wo_B_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_wo_B_qwen7b \
    --model_name intfloat/e5-large-v2

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_ours_wo_sr_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_ours_wo_sr_qwen7b \
    --model_name intfloat/e5-large-v2

!CUDA_VISIBLE_DEVICES=0 python scripts/build_index.py \
    --corpus_path /KidnapRAG/ReAct/datasets/hotpotqa/REACT_ours_wo_Chaining_qwen7b.jsonl \
    --output_dir /KidnapRAG/ReAct/datasets/hotpotqa/REACT_ours_wo_Chaining_qwen7b \
    --model_name intfloat/e5-large-v2

%cd /content/KidnapRAG/ReAct/ReAct
!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method wo_B \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method ours_wo_sr \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 python attack_react.py \
    --attack_method ours_wo_Chaining \
    --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
    --seed 1 \
    --dataset hotpotqa \
    --max_samples 10

# Webthinker - hotpotqa

%cd /content/KidnapRAG/Webthinker

!bash scripts/webthinker/qwen7b_clean.sh \
  ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
  ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
  ./results/hotpotqa/clean_qwen7b.json

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python scripts/generate_attack_doc_kill.py \
    --input_path /KidnapRAG/Webthinker/results/hotpotqa/clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type ours \
    --max_samples 10

!python concat.py --dataset hotpotqa --gen_model qwen7b

!head -n 200 ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BM_qwen7b.jsonl \
  > ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BM_qwen7b_small.jsonl

!head -n 200 ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCM_qwen7b.jsonl \
  > ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCM_qwen7b_small.jsonl

!wc -l ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BM_qwen7b_small.jsonl
!wc -l ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCM_qwen7b_small.jsonl

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BM_qwen7b_small.jsonl \
    ./results/hotpotqa/BM_qwen7b.json

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCM_qwen7b_small.jsonl \
    ./results/hotpotqa/BCM_qwen7b.json

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCCM_qwen7b.jsonl \
    ./results/hotpotqa/BCCM_qwen7b.json

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCCCM_qwen7b.jsonl \
    ./results/hotpotqa/BCCCM_qwen7b.json

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_BCCCCM_qwen7b.jsonl \
    ./results/hotpotqa/BCCCCM_qwen7b.json

## w/o B, M, Chaining (Table 2)

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python scripts/generate_attack_doc_kill.py \
    --input_path /KidnapRAG/Webthinker/results/hotpotqa/clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type wo_B \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python scripts/generate_attack_doc_kill.py \
    --input_path /KidnapRAG/Webthinker/results/hotpotqa/clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type ours_wo_sr \
    --max_samples 10

!CUDA_VISIBLE_DEVICES=0 \
    KIDNAPRAG_ATTACKER_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" \
    python scripts/generate_attack_doc_kill.py \
    --input_path /KidnapRAG/Webthinker/results/hotpotqa/clean_qwen7b.json \
    --gpu_num 1 \
    --dataset hotpotqa \
    --type ours_wo_Chaining \
    --max_samples 10

%cd /content/KidnapRAG/Webthinker
!PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python src/retrieval/build_index.py \
    --corpus-path /content/KidnapRAG/Webthinker/datasets/hotpotqa_poisoned_corpus/WEBTHINKER_wo_B_qwen7b.jsonl \
    --index-dir /content/KidnapRAG/Webthinker/datasets/hotpotqa_poisoned_corpus/WEBTHINKER_wo_B_qwen7b \
    --model-name intfloat/e5-large-v2

!PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python src/retrieval/build_index.py \
    --corpus-path /content/KidnapRAG/Webthinker/datasets/hotpotqa_poisoned_corpus/WEBTHINKER_ours_wo_sr_qwen7b.jsonl \
    --index-dir /content/KidnapRAG/Webthinker/datasets/hotpotqa_poisoned_corpus/WEBTHINKER_ours_wo_sr_qwen7b \
    --model-name intfloat/e5-large-v2

!PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python src/retrieval/build_index.py \
    --corpus-path /content/KidnapRAG/Webthinker/datasets/hotpotqa_poisoned_corpus/WEBTHINKER_ours_wo_Chaining_qwen7b.jsonl \
    --index-dir /content/KidnapRAG/Webthinker/datasets/hotpotqa_poisoned_corpus/WEBTHINKER_ours_wo_Chaining_qwen7b \
    --model-name intfloat/e5-large-v2

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_wo_B_qwen7b.jsonl \
    ./results/hotpotqa/wo_B_qwen7b.json

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_ours_wo_sr_qwen7b.jsonl \
    ./results/hotpotqa/ours_wo_sr_qwen7b.json

!bash scripts/webthinker/qwen7b_attack.sh \
    ./datasets/hotpotqa/hotpotqa_promptinject_select.json \
    ./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl \
    ./datasets/hotpotqa_poisoned_corpus/WEBTHINKER_ours_wo_Chaining_qwen7b.jsonl \
    ./results/hotpotqa/ours_wo_Chaining_qwen7b.json