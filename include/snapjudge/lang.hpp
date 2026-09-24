#pragma once
// snapjudge lang.hpp: dependency-free language/script detection, port of
// lang.py. All behavior (stopword lists, thresholds, margins) identical.

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;

// Flatten the string leaves of a state (str/dict/list), joined, capped.
std::string state_text(const ordered_json& state, int max_chars = 4000);

// Dominant script: "latin", "han", ... or "unknown" with no letters.
std::string detect_script(const std::string& text);

// Fraction of alphabetic chars per script.
std::unordered_map<std::string, double> script_profile(const std::string& text);

struct LatinProfile {
  std::string language;          // "" == None (undecided)
  int english_hits = 0;
  double diacritic_rate = 0.0;
  bool looks_non_english = false;
};
LatinProfile latin_profile(const std::string& text);
std::string guess_latin_language(const std::string& text);

// analyse() result: keys match the Python dict exactly:
//   script, script_profile, language (null possible), is_english,
//   language_undecided, diacritic_rate, non_latin_fraction
ordered_json analyse(const ordered_json& state);

bool is_english(const ordered_json& state);

}  // namespace snapjudge
