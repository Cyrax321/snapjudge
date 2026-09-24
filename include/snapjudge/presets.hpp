#pragma once
// snapjudge presets.hpp: ready-to-use question presets, port of presets.py.

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;

ordered_json triage_questions();
ordered_json email_questions();                                   // default categories
ordered_json email_questions(const ordered_json& categories);     // caller-supplied
ordered_json guard_questions();
ordered_json moderation_questions();
ordered_json router_questions();

}  // namespace snapjudge
