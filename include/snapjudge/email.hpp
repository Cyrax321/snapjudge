#pragma once
// snapjudge email.hpp: email cleaning/structuring, port of email.py.
// Python `re` semantics are preserved via PCRE2 with UTF+UCP (lookbehind and
// scoped (?i:...) groups both supported verbatim).

#include <string>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;

std::string clean_email_body(const std::string& body, int max_chars = 3000);

ordered_json email_state(const std::string& subject, const std::string& body,
                         const ordered_json& extra = ordered_json::object(),
                         const std::string& sender = "", bool clean = true);

}  // namespace snapjudge
