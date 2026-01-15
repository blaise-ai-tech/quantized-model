import torch
from safetensors.torch import load_file, save_file
from typing import Dict
import argparse
import os
import json

N_LAYERS = 61
N_HEADS = 128
Q_LORA_RANK = 1536
V_HEAD_DIM = 128
GATE_DIM = N_HEADS * V_HEAD_DIM

def create_gate_weights(index_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    new_shard_name = "gates-model-00066-of-00065.safetensors"

    gate_weights = {}
    new_weight_entries = {}

    for layer_idx in range(N_LAYERS):
        weight_key = f"model.layers.{layer_idx}.self_attn.g1_gate.weight"
        gate_weights[weight_key] = torch.zeros(GATE_DIM, Q_LORA_RANK, dtype=torch.bfloat16)
        new_weight_entries[weight_key] = new_shard_name

    shard_path = os.path.join(output_dir, new_shard_name)
    save_file(gate_weights, shard_path)

    updated_weight_map = dict(weight_map)
    updated_weight_map.update(new_weight_entries)
    index_data["weight_map"] = updated_weight_map

    gate_total_bytes = N_LAYERS * GATE_DIM * Q_LORA_RANK * 2
    if "metadata" in index_data and "total_size" in index_data["metadata"]:
        index_data["metadata"]["total_size"] += gate_total_bytes

    updated_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(updated_index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    size_mb = os.path.getsize(shard_path) / (1024**2)
    print(f"Created {new_shard_name} ({size_mb:.2f} MB)")
    print("Updated model.safetensors.index.json")


class GateWeightLoader:
    def __init__(self, gate_weights_path: str):
        self.gate_weights = load_file(gate_weights_path)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return self.gate_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=str, required=True, help="Path to model.safetensors.index.json")
    parser.add_argument("--output-dir", "-o", type=str, default="./gate_output")

    args = parser.parse_args()
    create_gate_weights(args.index, args.output_dir)
