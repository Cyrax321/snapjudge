#pragma once
// snapjudge shortlist.hpp: coarse-to-fine choice shortlisting, port of
// shortlist.py.

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;
class Agent;

using EmbedFn = std::function<std::vector<std::vector<double>>(const std::vector<std::string>&)>;

constexpr int DEFAULT_SHORTLIST_K = 20;

// Top-k choice labels for `state` by cosine similarity of embed_fn.
std::vector<std::string> shortlist_choice(const ordered_json& state,
                                          const ordered_json& criteria,
                                          const EmbedFn& embed_fn, int k = DEFAULT_SHORTLIST_K,
                                          const std::string& instructions = "");

// Shortlist each choice question then call the runner once. Result gets a
// "shortlist" key mirroring Python's predict_shortlist.
ordered_json predict_shortlist(
    const std::function<ordered_json(const ordered_json&, const ordered_json&)>& runner,
    const ordered_json& state, const ordered_json& questions, const EmbedFn& embed_fn,
    int k = DEFAULT_SHORTLIST_K);

// As predict_shortlist, but only shortlists choice questions whose option count
// EXCEEDS `threshold` (others pass through untouched). threshold=0 shortlists
// every choice question. This is the coarse-to-fine path for high-cardinality
// labels (e.g. Banking77's 77 intents) where a single-pass head starves each
// option of its token budget.
ordered_json predict_shortlist_above(
    const std::function<ordered_json(const ordered_json&, const ordered_json&)>& runner,
    const ordered_json& state, const ordered_json& questions, const EmbedFn& embed_fn,
    int k = DEFAULT_SHORTLIST_K, int threshold = DEFAULT_SHORTLIST_K);

// Mean-pool the resident checkpoint encoder (embed_fn_from_agent in Python).
EmbedFn embed_fn_from_agent(const Agent& agent, int max_length = 512, int batch_size = 32);

}  // namespace snapjudge
