# Arithmetic LLM Training System - Quick Start Summary

## Quick Start Demo: Full Pipeline from Beginning to End

Follow these commands to reproduce the complete training pipeline:

Run commands from the repository root (`stat359/`).

Shell compatibility:
- Bash/zsh multiline examples use `\` line continuation.
- PowerShell multiline examples use the backtick `` ` `` for continuation.
- Single-line fallback (works in both shells):
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --num-samples 100 --batch-size 1
```

Placeholder replacement:
- Replace `YYYYMMDD_HHMMSS` with an actual run directory under `models/`.
- Example resolved path: `models/foundational_20260201_012912_173614/best_model.pt`.

### Step 1: Corpus Generation
```bash
poetry run python -m instructor.final_project.arithmetic_llm.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt

poetry run python -m instructor.final_project.arithmetic_llm.generate_instruction_corpus_mixed \
  --num-samples 20000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.0 \
  --output-mixed data/instruction_corpus.txt

poetry run python -m instructor.final_project.arithmetic_llm.generate_corpus \
  --instruction-only \
  --num-samples 1000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.01 \
  --output-instruction data/instruction_corpus_test.txt
```

### Step 2: Tokenizer Training
```bash
poetry run python -m instructor.final_project.arithmetic_llm.train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000
```

### Step 3: Sequence Analysis
```bash
poetry run python -m instructor.final_project.arithmetic_llm.check_sequence_lengths \
  --corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --corpus-type instruction
```

### Step 4: Foundational Model Training
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --max-seq-length 512 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --num-epochs 5
```

### Step 5: Foundational Model Evaluation
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1
```

### Step 6: Instruction Fine-tuning
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 5 \
  --batch-size 16 \
  --learning-rate 0.0001
```

### Step 7: Instruction Model Evaluation
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

### Step 8: LoRA Fine-tuning
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training_lora \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 3 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --save-merged-model
```

### Step 9: LoRA Model Evaluation
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

### Step 10: GRPO Training (Reinforcement Learning)
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_grpo_training \
  --instruction-corpus data/instruction_corpus.txt \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models/grpo \
  --data-mode instruction \
  --num-epochs 3 \
  --batch-size 8 \
  --num-candidates 4 \
  --temperature 0.8 \
  --kl-penalty-coef 0.05
```

### Step 11: GRPO Model Evaluation
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/grpo/grpo_YYYYMMDD_HHMMSS/final_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

