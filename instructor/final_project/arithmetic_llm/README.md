# Arithmetic LLM Training System

A two-phase machine learning pipeline that trains a transformer-based language model to solve arithmetic problems with step-by-step reasoning.

See `NOTES.md` for consolidated technical notes on tokenizer behavior, data loading, and evaluation details.


## Overview

This system trains a language model to:
- Evaluate arithmetic expressions with addition (+) and subtraction (-)
- Provide detailed step-by-step reasoning
- Produce accurate final results

The training process consists of two phases:
1. **Foundational Training**: Train a base model on arithmetic expressions and evaluations
2. **Instruction Fine-tuning**: Fine-tune the model to respond to instruction prompts with structured reasoning

### Data Format

The system uses **JSONL (JSON Lines)** format for corpus data. Each line is a JSON object with structured fields:

```json
{
  "expression": "5 + (10 - 3)",
  "problem": "Evaluate: 5 + (10 - 3)",
  "solution": "<think>\nStep 1: 10 - 3 = 7\n...\n</think>\nFinal Result: 12",
  "answer": 12
}
```

**Field Descriptions:**
- `expression`: Raw arithmetic expression
- `problem`: Formatted problem statement with "Evaluate:" prefix
- `solution`: Complete solution with `<think>` tags and step-by-step reasoning
- `answer`: Final numeric result or "ERROR" for invalid expressions

**Training Modes:**

The data loader processes JSONL data differently based on training mode:

1. **Foundational Mode**: Concatenates `problem + solution` to train the base model on complete sequences
   - Input: `"Evaluate: 5 + (10 - 3) <think> Step 1: 10 - 3 = 7 ... Final Result: 12"`
   - Model learns when to generate `<think>` and complete reasoning

2. **Instruction Mode**: Uses `problem + " <think>"` as prompt, `solution` as target
   - Prompt: `"Evaluate: 5 + (10 - 3) <think>"`
   - Target: `"<think> Step 1: 10 - 3 = 7 ... Final Result: 12"`
   - Model learns to generate reasoning given instruction prompts
   - Loss computed only on target tokens (prompt tokens masked)


## Quick Start

Here's a complete workflow from corpus generation to interactive solving:

Run commands from the repository root (`stat359/`) and prefer the canonical form:
- `poetry run python -m instructor.final_project.arithmetic_llm.<module>`

Shell-safe command syntax:
- Bash/zsh multiline examples use `\`.
- PowerShell multiline examples use the backtick `` ` ``.
- Single-line fallback (works in Bash/zsh and PowerShell):
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt --tokenizer-path data/tokenizer --max-gen-length 512 --num-samples 100 --batch-size 1
```

PowerShell multiline example:
```powershell
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation `
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt `
  --tokenizer-path data/tokenizer `
  --max-gen-length 512 `
  --num-samples 100 `
  --batch-size 1
```

Placeholder replacement:
- Replace `YYYYMMDD_HHMMSS` with a real run directory under `models/`.
- Example resolved path: `models/foundational_20260201_012912_173614/best_model.pt`.

```bash
# 1. Generate training corpus (100,000 samples recommended)

# Generate foundational training corpus with 100K samples (plain text)
# This large corpus provides the base model with extensive arithmetic patterns
poetry run python -m instructor.final_project.arithmetic_llm.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt

# Generate mixed instruction corpus (valid + invalid)
# This creates a balanced dataset without writing intermediate files
poetry run python -m instructor.final_project.arithmetic_llm.generate_instruction_corpus_mixed \
  --num-samples 20000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0 \
  --output-mixed data/instruction_corpus.txt

# Generate separate test corpus for evaluation (1K samples, minimal errors)
# This provides a clean test set with only 1% invalid expressions
poetry run python -m instructor.final_project.arithmetic_llm.generate_corpus \
  --instruction-only \
  --num-samples 1000 \
  --max-depth 4 \
  --output-instruction data/instruction_corpus_test.txt \
  --num-range 1 20 \
  --invalid-rate 0

# check line counts (justified `-c` exception for quick inspection)
poetry run python -c "import sys; [print(f'{sum(1 for _ in open(f))} {f}') for f in ['data/foundational_corpus.txt', 'data/instruction_corpus.txt', 'data/instruction_corpus_test.txt']]"

# 2. Train tokenizer
poetry run python -m instructor.final_project.arithmetic_llm.train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000

# show tokenizer table
poetry run python -m instructor.final_project.arithmetic_llm.print_token_table --tokenizer_path data/tokenizer/tokenizer.pkl  > tokens.csv

# Analyze your instruction corpus
poetry run python -m instructor.final_project.arithmetic_llm.check_sequence_lengths \
  --corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer

# 3. Train foundational model
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --num-epochs 10 \
  --max-seq-length 512 \
  --batch-size 16

#3.1 Evaluate the foundational model, performance would be bad
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1


# 4. Fine-tune instruction model
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 10

# 4.1 Evaluate the model
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

# 5 Fine-tune with LoRA adapters (optional)
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training_lora \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 10 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --save-merged-model


# 5.1 Evaluate the LoRA merged model (optional)
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

# 6 GRPO training (optional)
poetry run python -m instructor.final_project.arithmetic_llm.run_grpo_training \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models/grpo \
  --data-mode generated \
  --log-every 1 \
  --num-samples 1024 \
  --num-epochs 3 \
  --num-candidates 8 \
  --max-gen-length 511 \
  --temperature 0.8 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --kl-penalty-coef 0.05
 

# 6.1 Evaluate GRPO model
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/grpo/grpo_YYYYMMDD_HHMMSS/final_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --batch-size 1 \
  --num-samples 1000

```

### Path and Artifact Rules

- `--model-path`, `--foundational-checkpoint`, and `--sft-checkpoint` expect checkpoint **files** (for example `best_model.pt` or `final_model.pt`), not directories.
- `--tokenizer-path` expects a tokenizer **directory** (for example `data/tokenizer`).
- If evaluating a LoRA adapter file (`lora_adapter.pt`), also provide `--base-checkpoint` or merge adapters first.

## Detailed Usage

### 1. Corpus Generation

Generate training data consisting of arithmetic expressions and their step-by-step evaluations.

```bash
# Foundational corpus (plain text)
poetry run python -m instructor.final_project.arithmetic_llm.generate_foundational_plaintext \
  --num-samples 50000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.1 \
  --output-txt data/foundational_corpus_plain.txt

# Mixed instruction corpus (valid + invalid)
poetry run python -m instructor.final_project.arithmetic_llm.generate_instruction_corpus_mixed \
  --num-samples 50000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.1 \
  --output-mixed data/instruction_corpus.txt
```

**Corpus Size Recommendations:**
- **Minimum (quick testing)**: 10,000 samples - Fast generation and training, but may underfit
- **Recommended**: 50,000 samples - Good balance of training time and model accuracy
- **Optimal**: 100,000+ samples - Best accuracy and generalization, longer training time

For production-quality results, use at least 50,000 samples. The model needs sufficient data to learn arithmetic patterns, order of operations, and step-by-step reasoning.

**Parameters:**
- `--num-samples`: Number of expression-evaluation pairs to generate (required)
- `--max-depth`: Maximum depth of expression trees (default: 5)
- `--num-range`: Range of numbers to use (default: 1 20)
- `--invalid-rate`: Fraction of invalid expressions for robustness (default: 0.1)
- `--output-foundational`: Path for foundational corpus
- `--output-instruction`: Path for instruction corpus
- `--foundational-only`: Generate only foundational corpus
- `--instruction-only`: Generate only instruction corpus

**Mixed Instruction Corpus Script:**
- `poetry run python -m instructor.final_project.arithmetic_llm.generate_instruction_corpus_mixed` creates a mixed instruction corpus
- `--output-mixed`: Path for the mixed instruction corpus

**Foundational Plain-Text Script:**
- `poetry run python -m instructor.final_project.arithmetic_llm.generate_foundational_plaintext` generates shuffled plain text directly
- `--output-txt`: Path to plain-text corpus for training

**Output Format:**

Foundational corpus (post-processed, plain text):
```
Evaluate: 5 + (10 - 3)
<think> Step 1: 10 - 3 = 7 Expression now: 5 + 7 Step 2: 5 + 7 = 12 Expression now: 12 </think> Final Result: 12
```

Instruction corpus:
```
Evaluate: 5 + (10 - 3) <think>
<think>
Step 1: 10 - 3 = 7
Expression now: 5 + 7
Step 2: 5 + 7 = 12
Expression now: 12
</think>
Final Result: 12
```

### 2. Tokenizer Training

Train a BPE (Byte Pair Encoding) tokenizer on the arithmetic corpus.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.train_tokenizer \
  --corpus-path data/foundational_corpus_plain.txt \
  --vocab-size 1000 \
  --output-dir data/tokenizer
```

**Parameters:**
- `--corpus-path`: Path to training corpus (required)
- `--vocab-size`: Target vocabulary size (default: 1000)
- `--output-dir`: Directory to save tokenizer (default: data/tokenizer)

**Special Tokens:**
- `<pad>`: Padding token
- `<unk>`: Unknown token
- `<bos>`: Beginning of sequence
- `<eos>`: End of sequence
- `<think>`: Start of reasoning
- `</think>`: End of reasoning

### 3. Foundational Model Training

Train the base transformer model on arithmetic expressions.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/foundational_corpus_plain.txt \
  --tokenizer-path data/tokenizer \
  --output-dir models \
  --num-epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-4
```

**Training Parameters:**
- `--corpus-path`: Path to training corpus (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--output-dir`: Directory to save checkpoints (default: models)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 10)
- `--warmup-steps`: Warmup steps for learning rate (default: 1000)
- `--gradient-clip`: Gradient clipping value (default: 1.0)
- `--save-every`: Save checkpoint every N steps (default: 1000)
- `--device`: Device for training: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)

**Model Architecture Parameters:**
- `--d-model`: Embedding dimension (default: 256)
- `--nhead`: Number of attention heads (default: 8)
- `--num-layers`: Number of transformer layers (default: 6)
- `--dim-feedforward`: Feedforward dimension (default: 1024)
- `--dropout`: Dropout rate (default: 0.1)

**Configuration Files:**
You can also use JSON configuration files:
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/foundational_corpus_plain.txt \
  --tokenizer-path data/tokenizer \
  --config training_config.json \
  --model-config model_config.json
```

**Output:**
Training creates a timestamped directory containing:
- `best_model.pt`: Best model checkpoint (lowest validation loss)
- `final_model.pt`: Final model checkpoint
- `checkpoint_step_N.pt`: Intermediate checkpoints
- `training_config.json`: Training configuration
- `model_config.json`: Model architecture configuration
- `training_log.json`: Training metrics per epoch
- `training_summary.json`: Final training summary

### 4. Instruction Fine-tuning

Fine-tune the foundational model with instruction-formatted data.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models \
  --num-epochs 5 \
  --batch-size 32 \
  --learning-rate 5e-5
```

**Parameters:**
- `--instruction-corpus-path`: Path to instruction corpus (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--foundational-checkpoint`: Path to foundational model checkpoint (required)
- `--output-dir`: Directory to save checkpoints (default: models)
- `--learning-rate`: Learning rate (default: 5e-5, lower than foundational)
- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 5)
- `--warmup-steps`: Warmup steps (default: 500)
- `--gradient-clip`: Gradient clipping (default: 1.0)
- `--save-every`: Save checkpoint every N steps (default: 500)
- `--device`: Device for training: 'cuda', 'mps', 'cpu', or 'auto' (default: auto)

**Note:** Fine-tuning typically uses a lower learning rate and fewer epochs than foundational training.

### 4.1 LoRA Instruction Fine-tuning

Fine-tune with LoRA adapters for parameter-efficient training.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training_lora \
  --instruction-corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models \
  --num-epochs 3 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --lora-dropout 0.0 \
  --save-merged-model
```

**Parameters (LoRA-specific):**
- `--lora-rank`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha scaling (default: 16.0)
- `--lora-target-modules`: Comma-separated target modules (attention, feedforward)
- `--lora-dropout`: LoRA dropout rate (default: 0.0)
- `--save-merged-model`: Save merged model for inference

**Outputs:**
- `lora_adapter.pt`: LoRA adapter weights
- `merged_model.pt`: Optional merged model when `--save-merged-model` is used

### 4.2 GRPO Training

Train with Group Relative Policy Optimization (GRPO) using verifiable rewards.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_grpo_training \
  --instruction-corpus data/instruction_corpus.txt \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models/grpo \
  --num-epochs 3 \
  --batch-size 8 \
  --num-candidates 4 \
  --temperature 0.8 \
  --kl-penalty-coef 0.05
```

**Parameters:**
- `--instruction-corpus`: Instruction corpus JSONL (required for instruction mode)
- `--tokenizer`: Tokenizer directory (required)
- `--sft-checkpoint`: SFT checkpoint path (required)
- `--output-dir`: Output directory for checkpoints and logs
- `--data-mode`: `instruction` or `generated` (default: instruction)
- `--num-candidates`: Candidates per prompt (default: 4)
- `--kl-penalty-coef`: KL penalty coefficient (default: 0.05)
- `--temperature`, `--top-k`, `--top-p`: Sampling parameters
- `--gradient-accumulation-steps`: Accumulate gradients across steps

**Outputs:**
- `checkpoint_step_N.pt`: Periodic checkpoints
- `best_model.pt`: Best checkpoint by validation reward rate (if validation enabled)
- `final_model.pt`: Final checkpoint
- `grpo_training_log.json`: Training metrics log

### 5. Model Evaluation

Evaluate the trained model on a test set of arithmetic expressions.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000 \
  --max-depth 4 \
  --output-dir evaluation_results
```

**LoRA Evaluation (merged model):**
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --num-samples 1000 \
  --max-depth 4 \
  --output-dir evaluation_results
```

If you saved only adapters, merge them into a base checkpoint first:
```bash
poetry run python -m instructor.final_project.arithmetic_llm.merge_lora_adapter \
  --base-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --adapter-path models/instruction_lora_YYYYMMDD_HHMMSS/lora_adapter.pt \
  --output-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt
```

**Parameters:**
- `--model-path`: Path to model checkpoint (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--num-samples`: Number of test expressions (default: 1000)
- `--max-depth`: Maximum expression depth (default: 5)
- `--num-range`: Number range for test expressions (default: 1 20)
- `--output-dir`: Directory to save results (default: evaluation_results)
- `--device`: Device for inference (default: auto)

**Metrics:**
- **Exact Match Accuracy**: Percentage of correct final results
- **Parse Success Rate**: Percentage of parseable outputs
- **Average Generation Length**: Mean number of tokens generated

**Output:**
Evaluation creates timestamped files:
- `evaluation_metrics_YYYYMMDD_HHMMSS.json`: Detailed metrics
- `sample_outputs_YYYYMMDD_HHMMSS.json`: Sample model outputs
- `evaluation_summary_YYYYMMDD_HHMMSS.txt`: Human-readable summary

**Expected Performance:**
- Good models: 60-80% accuracy
- Excellent models: 80%+ accuracy
- Parse success rate should be >90%

### 6. Interactive Solver

Use the trained model interactively to solve arithmetic problems.

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_interactive \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer
```

**Parameters:**
- `--model-path`: Path to instruction-tuned model (required)
- `--tokenizer-path`: Path to tokenizer directory (required)
- `--device`: Device for inference (default: auto)

**Usage:**
```
Enter expression: 5 + (10 - 3)

------------------------------------------------------------
SOLUTION:
------------------------------------------------------------

Reasoning Steps:
  Step 1: 10 - 3 = 7
  Expression now: 5 + 7
  Step 2: 5 + 7 = 12
  Expression now: 12

Final Result: 12
------------------------------------------------------------

Enter expression: exit
```

**Commands:**
- Type any arithmetic expression to solve it
- Type `exit`, `quit`, or `q` to exit
- Press `Ctrl+C` to exit



## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:
1. Reduce batch size: `--batch-size 16` or `--batch-size 8`
2. Reduce model size: `--d-model 128 --num-layers 4`
3. Use CPU: `--device cpu` (slower but uses system RAM)

### Low Accuracy

If model accuracy is low:
1. Generate more training data: `--num-samples 100000` (increase to 100K or more)
2. Train for more epochs: `--num-epochs 20`
3. Increase model size: `--d-model 512 --num-layers 8`
4. Verify corpus quality: Check that expressions are valid and diverse
5. Ensure sufficient corpus size: At least 50K samples for foundational training

### Tokenizer Issues

If tokenizer produces unexpected results:
1. Increase vocabulary size: `--vocab-size 2000`
2. Verify corpus contains diverse expressions
3. Check that special tokens are preserved

### Training Not Converging

If training loss doesn't decrease:
1. Reduce learning rate: `--learning-rate 5e-5`
2. Increase warmup steps: `--warmup-steps 2000`
3. Check gradient clipping: `--gradient-clip 0.5`
4. Verify data quality and format

## Advanced Usage

### Custom Training Configuration

Create a JSON configuration file for reproducible training:

```json
{
  "learning_rate": 1e-4,
  "batch_size": 32,
  "num_epochs": 10,
  "warmup_steps": 1000,
  "gradient_clip": 1.0,
  "save_every": 1000,
  "eval_every": 500,
  "device": "cuda"
}
```

Use it with:
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/corpus.txt \
  --tokenizer-path data/tokenizer \
  --config training_config.json
```

### Custom Model Architecture

Create a model configuration file:

```json
{
  "d_model": 512,
  "nhead": 8,
  "num_layers": 8,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "max_seq_length": 512
}
```

Use it with:
```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/corpus.txt \
  --tokenizer-path data/tokenizer \
  --model-config model_config.json
```

### Resuming Training

To resume training from a checkpoint, use the checkpoint as the starting point for a new training run. Note that this starts a new training session rather than continuing the previous one.

### Distributed Training

Current behavior:
- `--device auto` selects `cuda` first, then `mps`, then `cpu`.
- Training and evaluation run on a single selected device only (no built-in multi-GPU parallelism).

DDP extension map (files/modules to modify):
- Entry points: `instructor/final_project/arithmetic_llm/run_foundational_training.py`, `instructor/final_project/arithmetic_llm/run_instruction_training.py`, `instructor/final_project/arithmetic_llm/run_instruction_training_lora.py`, `instructor/final_project/arithmetic_llm/run_grpo_training.py`.
- Training loops: `instructor/final_project/arithmetic_llm/train_foundational.py`, `instructor/final_project/arithmetic_llm/train_instruction.py`, `instructor/final_project/arithmetic_llm/train_instruction_lora.py`, `instructor/final_project/arithmetic_llm/grpo_trainer.py`.
- Sampler/dataloader wiring: `instructor/final_project/arithmetic_llm/data_loader.py` (use distributed samplers and rank-aware shuffling).
- Checkpointing and resume: `instructor/final_project/arithmetic_llm/train_foundational.py` (`save_checkpoint` / `load_checkpoint`), `instructor/final_project/arithmetic_llm/grpo_trainer.py` (`save_checkpoint` / `load_checkpoint`), plus resume handling in instruction training scripts.

### Runtime Compatibility

| Component | Verified/Expected | Notes |
|---|---|---|
| Python | 3.10+ | Project is managed with Poetry (`python = "^3.10"`). |
| PyTorch | 2.7.1 (from lockfile) | Use Poetry-managed environment to avoid wheel mismatch. |
| CUDA | Match your installed torch build | If CUDA runtime does not match, use CPU/MPS fallback or reinstall a compatible torch wheel. |
| GPU architecture | CUDA, Apple MPS, CPU supported by code path | `--device auto` falls back to MPS/CPU when CUDA is unavailable. |

Known issue (Blackwell / RTX 50-series):
- The current locked torch build does not support Blackwell GPUs.
- Mitigations:
  1. Install a torch build that explicitly supports your CUDA/driver stack for Blackwell.
  2. Run with `--device cpu` (or `--device mps` on Apple Silicon) until a compatible CUDA wheel is installed.
  3. Use a known-compatible cloud GPU/runtime for training if local compatibility is blocked.

## Performance Tips

1. **Use GPU**: Training on GPU is 10-100x faster than CPU
2. **Batch Size**: Larger batches train faster but use more memory
3. **Corpus Size**: More data (50K-100K samples) improves accuracy
4. **Model Size**: Larger models (d_model=512, num_layers=8) are more accurate but slower
5. **Checkpointing**: Save checkpoints frequently to avoid losing progress


## Acknowledgments

- Transformer architecture adapted from the TinyStories project
- BPE tokenization based on HuggingFace tokenizers library
- Expression generation and evaluation utilities from existing codebase
