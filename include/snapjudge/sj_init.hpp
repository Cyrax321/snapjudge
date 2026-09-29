#pragma once
// snapjudge sj_init.hpp: create a fresh random-init native snapjudge (SjModel)
// checkpoint, self-contained (no Python). Writes sj_config.json,
// model.safetensors (random small weights, LayerNorm = 1 / bias = 0), and a
// byte-level tokenizer with the special ids the config expects.

#include <cstdint>
#include <string>

namespace snapjudge {

// Create a fresh checkpoint at `dir`. Shape parameters mirror sj_config.json.
// The tokenizer is a byte-level 512-vocab (256 GPT-2 byte symbols + specials
// + fillers), matching cls=1/sep=2/pad=3/mask=4/unk=0.
void init_sj_checkpoint(const std::string& dir, int vocab_size, int hidden_size,
                        int num_layers, int num_heads, int intermediate_size,
                        int max_positions, int head_layers, uint64_t seed);

}  // namespace snapjudge
