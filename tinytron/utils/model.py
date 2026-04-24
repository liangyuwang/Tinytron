import torch

from tinytron.model.config import ModelConfig

def get_model_params(model_config: ModelConfig):
    if model_config.use_moe:
        return get_moe_model_params(
            num_layer=model_config.num_layer,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            vocab_size=model_config.vocab_size,
            num_expert=model_config.num_experts,
            top_k=model_config.num_experts_per_tok if isinstance(model_config.num_experts_per_tok, int) else max(model_config.num_experts_per_tok),
            moe_intermediate_size=model_config.moe_intermediate_size,
        )
    else:
        return get_dense_model_params(
            num_layer=model_config.num_layer,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            vocab_size=model_config.vocab_size,
        )

def get_dense_model_params(
    num_layer: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
):
    """
    Compute parameter counts for a dense Transformer model.
    Args:
        num_layer (int): number of transformer layers (L)
        hidden_size (int): hidden dimension (H)
        intermediate_size (int): feed-forward intermediate size (I)
        vocab_size (int): vocabulary size (V)
    Returns:
        dict: total parameter counts in billions
    """
    H = hidden_size
    L = num_layer
    I = intermediate_size
    V = vocab_size

    # Embedding + tied output head
    P_embed = V * H

    # Dense transformer layer parameters:
    # Attention: ~4*H^2 (QKV + out projection)
    # FFN: ~3*H*I (three linear layers: H->I and I->H)
    P_dense_layer = 4 * H * H + 3 * H * I
    P_dense_all = L * P_dense_layer

    # Total parameters
    P_total = P_embed + P_dense_all

    return {
        "total_params_B": P_total / 1e9,
        "dense_params_B": P_total / 1e9,
    }

def get_moe_model_params(
    num_layer: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_expert: int,
    top_k: int,
    moe_intermediate_size: int,
):
    """
    Compute parameter counts for a Transformer model with MoE layers.
    
    Args:
        num_layer (int): number of transformer layers (L)
        hidden_size (int): hidden dimension (H)
        intermediate_size (int): feed-forward intermediate size (I)
        vocab_size (int): vocabulary size (V)
        num_expert (int): number of experts per MoE layer (E)
        top_k (int): number of experts activated per token (k)
        moe_intermediate_size (int): intermediate size for MoE experts (I_moe)
    
    Returns:
        dict: total and active parameter counts in billions
    """

    H = hidden_size
    L = num_layer
    I = intermediate_size
    E = num_expert
    k = top_k
    I_moe = moe_intermediate_size
    V = vocab_size

    # Embedding + tied output head
    P_embed = V * H

    # Dense transformer layer parameters:
    # Attention: ~4*H^2 (QKV + out projection)
    # FFN: ~2*H*I (two linear layers: H->I and I->H)
    P_dense_layer = 4 * H * H + 2 * H * I
    P_dense_all = L * P_dense_layer

    # Router parameters per layer: H * E
    P_router_layer = H * E
    P_router_all = L * P_router_layer

    # MoE experts:
    # Each expert: 3 * H * I_moe  (gate_proj + up_proj + down_proj)
    P_expert = 3 * H * I_moe
    P_moe_all = L * E * P_expert

    # Total parameters
    P_total = P_embed + P_dense_all + P_router_all + P_moe_all

    # Active MoE parameters per forward:
    # k experts activated per layer
    P_moe_active = L * k * P_expert

    # Active total
    P_active = P_embed + P_dense_all + P_router_all + P_moe_active

    return {
        "total_params_B": P_total / 1e9,
        "active_params_B": P_active / 1e9,
        "dense_params_B": (P_embed + P_dense_all + P_router_all) / 1e9,
        "moe_total_B": P_moe_all / 1e9,
        "moe_active_B": P_moe_active / 1e9,
    }


def get_compiled_to_uncompiled_mapping(raw_model, compiled_keys):
    """
    Creates a mapping dictionary from compiled key names to the model's parameter tensors.

    Args:
        raw_model (torch.nn.Module): The uncompiled model instance.
        compiled_keys (set or list): A collection of key names read from the checkpoint,
                                      which may have the '_orig_mod.' prefix.

    Returns:
        dict: A dictionary where keys are the compiled names and values are the
              corresponding parameter tensors from the model.
    """
    uncompiled_state_dict = raw_model.state_dict()
    uncompiled_keys = set(uncompiled_state_dict.keys())
    
    mapping = {}
    prefix = "_orig_mod."
    
    for compiled_key in compiled_keys:
        # Try to remove the prefix to get the expected uncompiled key
        if compiled_key.startswith(prefix):
            uncompiled_key = compiled_key[len(prefix):]
        else:
            uncompiled_key = compiled_key
            
        # If this uncompiled key actually exists in the current model
        if uncompiled_key in uncompiled_keys:
            # Create the mapping: {compiled_key: tensor_in_the_model}
            mapping[compiled_key] = uncompiled_state_dict[uncompiled_key]
        else:
            # This is a warning, indicating that a key from the checkpoint
            # could not be matched in the current model. This might happen if
            # the model architecture has truly changed or if there's another prefix we didn't account for.
            print(f"Warning: Could not find a match for checkpoint key '{compiled_key}' in the model.")
            
    # Check if any model parameters were not mapped
    # This helps detect if the checkpoint is missing keys that the model expects.
    mapped_uncompiled_keys = {k[len(prefix):] if k.startswith(prefix) else k for k in mapping.keys()}
    missing_in_ckpt = uncompiled_keys - mapped_uncompiled_keys
    if missing_in_ckpt:
        print("Warning: The following model parameters were not found in the checkpoint:")
        # Print only the first few to avoid spamming the console
        for key in sorted(list(missing_in_ckpt))[:5]:
            print(f"  - {key}")

    return mapping
