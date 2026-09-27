#pragma once
// snapjudge email.hpp: email cleaning and structuring.
//
// `clean_email_body` strips the noise that corrupts an inbound email before a
// decision model sees it: quoted history, signatures, device footers and
// confidentiality disclaimers. It keeps the *new* message and drops everything
// that a mail client appended after it. Markers cover English, Portuguese and
// Spanish clients. Implemented over PCRE2 with UTF+UCP so lookbehind and
// scoped (?i:...) groups work as in Python's `re`.

#include <string>

#include <nlohmann/json.hpp>

namespace snapjudge {

using nlohmann::ordered_json;

// Remove quoted history, signatures and disclaimers from an email body.
// Returns at most max_chars code points.
std::string clean_email_body(const std::string& body, int max_chars = 3000);

// Build a state object for email classification: {subject, body, from?...}.
// `extra` merges additional non-null fields; `clean=false` skips body cleaning.
ordered_json email_state(const std::string& subject, const std::string& body,
                         const ordered_json& extra = ordered_json::object(),
                         const std::string& sender = "", bool clean = true);

}  // namespace snapjudge
