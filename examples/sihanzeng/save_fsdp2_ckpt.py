"""Save Gemma4-IT model as FSDP2 checkpoint for fast loading in future jobs."""
import os, sys, logging, torch
logging.basicConfig(level=logging.INFO)

AUTOMODEL_DIR = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/Automodel"
sys.path.insert(0, AUTOMODEL_DIR)
sys.path.insert(0, "/opt/megatron-lm/megatron/core/distributed/fsdp/src")

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig

HF_MODEL = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/models/gemma-4-26B-A4B-it"
SAVE_DIR = "/mnt/lustre/metavmds0lstre/checkpoints/sihanzeng/nemo_ckpts/gemma4-it-fsdp2"

def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    if rank == 0:
        print(f"Loading {HF_MODEL} on {world_size} GPUs...")

    backend = BackendConfig(
        attn="te",
        linear="te",
        rms_norm="te",
        rope_fusion=True,
        experts="torch_mm",
        dispatcher="deepep",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )

    model = NeMoAutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        backend=backend,
    )

    if rank == 0:
        print(f"Model loaded. Saving FSDP2 checkpoint to {SAVE_DIR}...")

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Use torch distributed checkpoint save
    from torch.distributed.checkpoint import save
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    state_dict = get_model_state_dict(model)
    save(state_dict, checkpoint_id=SAVE_DIR)

    if rank == 0:
        # Also save tokenizer and config for future use
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        tokenizer.save_pretrained(os.path.join(SAVE_DIR, "tokenizer"))
        config = AutoConfig.from_pretrained(HF_MODEL, trust_remote_code=True)
        config.save_pretrained(os.path.join(SAVE_DIR, "config"))
        print(f"Checkpoint saved to {SAVE_DIR}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
