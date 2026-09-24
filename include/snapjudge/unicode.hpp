#pragma once
// snapjudge unicode.hpp: UTF-8 + Unicode property + NFC support, dependency-free.
// Tables are code-generated at build time by snapjudge/python/gen_unicode.py.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace snapjudge {

// ---- UTF-8 -----------------------------------------------------------------
// Decode one code point starting at s[i]; on malformed input yields U+FFFD and
// advances by one byte (same tolerance as Python's errors='replace').
char32_t utf8_decode_next(const std::string& s, size_t& i);
std::u32string utf8_decode(const std::string& s);
std::string utf8_encode(char32_t cp);
std::string utf8_encode(const std::u32string& cps);

// ---- properties (binary search over generated tables) -----------------------
bool uni_isalpha(char32_t cp);   // str.isalpha()   (Python semantics)
bool uni_isletter(char32_t cp);  // \p{L}           (general category L*)
bool uni_isnumber(char32_t cp);  // \p{N}           (Nd | Nl | No)
bool uni_isdigit(char32_t cp);   // category Nd
bool uni_isspace(char32_t cp);   // str.isspace()  (Python semantics)
bool uni_isws(char32_t cp);      // White_Space property (Rust regex \s, tokenizer)
bool uni_isword(char32_t cp);    // [^\W\d_] under re.UNICODE (letter or mark),
                                 //    matching lang.py's _WORD regex
bool uni_iswordchar(char32_t cp);  // Python re \w (letters, digits, marks, '_')
bool uni_iscombining(char32_t cp);

// ---- case -------------------------------------------------------------------
std::u32string uni_lower(const std::u32string& s);  // str.lower() (full case)
std::u32string uni_upper(const std::u32string& s);
char32_t uni_lower1(char32_t cp);  // simple 1-char lower; cp unchanged if none
bool uni_isupper(char32_t cp);     // str.isupper(): has cased+uppercase form

// ---- normalization ----------------------------------------------------------
std::u32string nfc(const std::u32string& s);
std::string nfc(const std::string& utf8);

}  // namespace snapjudge
