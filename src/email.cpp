#include "snapjudge/email.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "snapjudge/unicode.hpp"

namespace snapjudge {

namespace {

// Thin PCRE2 wrapper: search = any-position (Python re.search). All patterns
// compile with UTF+UCP. The code handle is a shared_ptr so Re is safely
// copyable into the static pattern vectors.
struct Re {
  std::shared_ptr<pcre2_code> code;
  Re(const char* pattern, bool caseless = false) {
    int errnum;
    PCRE2_SIZE erroff;
    uint32_t opts = PCRE2_UTF | PCRE2_UCP;
    if (caseless) opts |= PCRE2_CASELESS;
    pcre2_code* c = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern),
                                  PCRE2_ZERO_TERMINATED, opts, &errnum, &erroff, nullptr);
    if (!c) {
      fprintf(stderr, "snapjudge: PCRE2 compile failed for /%s/ err=%d@%zu\n",
              pattern, errnum, static_cast<size_t>(erroff));
      abort();
    }
    code.reset(c, pcre2_code_free);
  }
  bool search(const std::string& s) const {
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code.get(), nullptr);
    int rc = pcre2_match(code.get(), reinterpret_cast<PCRE2_SPTR>(s.c_str()), s.size(), 0, 0,
                         md, nullptr);
    pcre2_match_data_free(md);
    return rc >= 0;
  }
  // Return the matched span of group 0; npos if no match.
  std::pair<size_t, size_t> match_span(const std::string& s, size_t start = 0) const {
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(code.get(), nullptr);
    int rc = pcre2_match(code.get(), reinterpret_cast<PCRE2_SPTR>(s.c_str()), s.size(),
                         start, 0, md, nullptr);
    if (rc < 0) {
      pcre2_match_data_free(md);
      return {std::string::npos, std::string::npos};
    }
    size_t* ov = pcre2_get_ovector_pointer(md);
    std::pair<size_t, size_t> out{ov[0], ov[1]};
    pcre2_match_data_free(md);
    return out;
  }
};

const std::vector<Re>& quote_headers() {
  static const std::vector<Re> v = [] {
    std::vector<Re> t;
    t.emplace_back(R"(^\s*On .{0,300}wrote:\s*$)", true);
    t.emplace_back(R"(^\s*Em (?=.*\d).{0,300}escreveu:\s*$)", true);
    t.emplace_back(R"(^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$)", true);
    t.emplace_back(R"(^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,})", true);
    t.emplace_back(R"(^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,})", true);
    t.emplace_back(R"(^\s*_{8,}\s*$)");
    t.emplace_back(R"(^\s*From:\s.+$)", true);
    t.emplace_back(R"(^\s*De:\s.*[@<])", true);
    return t;
  }();
  return v;
}
const Re& attribution_tail() {
  static const Re r(R"(^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$)", true);
  return r;
}
const Re& attribution_head() {
  static const Re r(R"(^\s*(On|Em|El) (?=.*\d))", true);
  return r;
}
const Re& header_from_name() {
  static const Re r(R"(^\s*De:\s+\S)", true);
  return r;
}
const Re& header_next() {
  static const Re r(R"(^\s*(Enviad[oa]( em| el)?:\s|(Data|Fecha):\s.*\d{4}))", true);
  return r;
}

const std::vector<Re>& signature_markers() {
  static const std::vector<Re> v = [] {
    std::vector<Re> t;
    t.emplace_back(R"(^\s*--\s*$)");
    // scoped-case closing/name pattern from email.py (PCRE2 supports (?i:...)
    t.emplace_back(
        R"(^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?[\s,;:!.]*(?:[^\W\d_a-z\xDF-\xF6\xF8-\xFF][\w'-]*[\s,.]*){0,3}$)");
    t.emplace_back(R"(^\s*sent from my (iphone|android|mobile|ipad))", true);
    t.emplace_back(
        R"(^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$)",
        true);
    return t;
  }();
  return v;
}

const Re& device_footer() {
  static const Re r(
      R"(^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?) (iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows)( (iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows|para|for|no|na|\d+))*|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$)",
      true);
  return r;
}

const Re& disclaimer() {
  static const Re r(
      R"((\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|disclos|unauthori[sz]ed)|)"
      R"(\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|)"
      R"(if you (have )?received this (e-?mail|message) in error|)"
      R"(\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|)"
      R"(\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}(destinatári|destinatari|pessoa|persona|entidade|entidad)|)"
      R"(\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|)"
      R"(\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|)"
      R"(\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|)"
      R"(\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir))",
      true);
  return r;
}

const Re& sentence_split_re() {
  static const Re r(R"((?<=[.!?])\s+)");
  return r;
}

std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\n') {
      out.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  out.push_back(s.substr(start));
  return out;
}

std::string rstrip(std::string s) {
  while (!s.empty() &&
         std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}
std::string strip(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}
std::string lstrip(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  return s.substr(a);
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
  std::string out;
  bool first = true;
  for (const auto& p : v) {
    if (!first) out += sep;
    out += p;
    first = false;
  }
  return out;
}

bool starts_with_gt(const std::string& s) {
  return !lstrip(s).empty() && lstrip(s)[0] == '>';
}

// Python re.split with the (?<=[.!?])\s+ pattern: split AFTER a sentence
// terminator, keeping it attached to the preceding piece.
std::vector<std::string> sentence_split(const std::string& s) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (true) {
    auto [a, b] = sentence_split_re().match_span(s, pos);
    if (a == std::string::npos) {
      out.push_back(s.substr(pos));
      break;
    }
    out.push_back(s.substr(pos, a - pos));
    pos = b;  // skip the matched whitespace
  }
  return out;
}

bool starts_new_sentence(const std::string& line) {
  for (char32_t cp : utf8_decode(line)) {
    if (uni_isalpha(cp)) {
      std::u32string lo = uni_lower(std::u32string(1, cp));
      return !(lo.size() == 1 && lo[0] == cp);  // isupper: lower() changed it
    }
  }
  return false;
}

std::vector<std::string> split_fused_lines(const std::string& sentence) {
  if (sentence.find('\n') == std::string::npos) return {sentence};
  std::vector<std::string> pieces;
  std::string buf;
  for (const auto& ln0 : split_lines(sentence)) {
    std::string line = strip(ln0);
    if (line.empty()) continue;
    if (!buf.empty() && starts_new_sentence(line)) {
      pieces.push_back(buf);
      buf = line;
    } else {
      buf = buf.empty() ? line : buf + " " + line;
    }
  }
  if (!buf.empty()) pieces.push_back(buf);
  return pieces;
}

std::string strip_disclaimer(const std::string& paragraph) {
  if (!disclaimer().search(paragraph)) return paragraph;
  std::vector<std::string> parts;
  for (const auto& p : sentence_split(paragraph)) {
    std::string t = strip(p);
    if (!t.empty()) parts.push_back(t);
  }
  std::vector<std::string> pieces;
  for (const auto& p : parts) {
    if (disclaimer().search(p)) {
      for (const auto& q : split_fused_lines(p)) pieces.push_back(q);
    } else {
      pieces.push_back(p);
    }
  }
  std::vector<std::string> keep;
  for (const auto& p : pieces)
    if (!disclaimer().search(p)) keep.push_back(p);
  return join(keep, " ");
}

}  // namespace

std::string clean_email_body(const std::string& body, int max_chars) {
  std::string text = body;
  // (body or "").replace("\r\n","\n").replace("\r","\n").replace("\\n","\n")
  auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  replace_all(text, "\r\n", "\n");
  replace_all(text, "\r", "\n");
  replace_all(text, "\\n", "\n");

  std::vector<std::string> lines;
  std::vector<std::string> src = split_lines(text);
  for (size_t i = 0; i < src.size(); ++i) {
    const std::string& line = src[i];
    bool is_header = false;
    for (const auto& re : quote_headers())
      if (re.search(line)) { is_header = true; break; }
    if (is_header && !lines.empty()) break;
    if (!lines.empty() && header_from_name().search(line) && i + 1 < src.size() &&
        header_next().search(src[i + 1]))
      break;
    if (attribution_tail().search(line) && !lines.empty()) {
      if (attribution_head().search(lines.back())) lines.pop_back();
      break;
    }
    if (starts_with_gt(line)) continue;
    lines.push_back(rstrip(line));
  }

  size_t cut = lines.size();
  size_t scan_from = std::max<size_t>(1, std::min(static_cast<size_t>(lines.size() * 0.6),
                                                  lines.size() >= 8 ? lines.size() - 8 : 0));
  for (size_t i = scan_from; i < lines.size(); ++i) {
    std::string t = strip(lines[i]);
    size_t nch = utf8_decode(t).size();
    if ((nch <= 40 &&
         std::any_of(signature_markers().begin(), signature_markers().end(),
                     [&](const Re& re) { return re.search(lines[i]); })) ||
        (nch <= 60 && device_footer().search(lines[i]))) {
      cut = i;
      break;
    }
  }
  lines.resize(cut);

  // split on blank lines (\n\s*\n), normalize
  std::string joined = join(lines, "\n");
  std::vector<std::string> paragraphs;
  {
    // Python re.split(r"\n\s*\n", text)
    size_t pos = 0;
    static const Re para_re(R"(\n\s*\n)");
    while (true) {
      auto [a, b] = para_re.match_span(joined, pos);
      if (a == std::string::npos) {
        paragraphs.push_back(joined.substr(pos));
        break;
      }
      paragraphs.push_back(joined.substr(pos, a - pos));
      pos = b;
    }
  }
  std::vector<std::string> cleaned;
  for (const auto& p : paragraphs) {
    std::string s = strip(strip_disclaimer(p));
    if (!s.empty()) cleaned.push_back(s);
  }
  std::string out = join(cleaned, "\n\n");
  // re.sub(r"[ \t]+", " ", ...)
  {
    static const Re ws_re(R"([ \t]+)");
    std::string norm;
    size_t pos = 0;
    while (true) {
      auto [a, b] = ws_re.match_span(out, pos);
      if (a == std::string::npos) {
        norm += out.substr(pos);
        break;
      }
      norm += out.substr(pos, a - pos);
      norm += " ";
      pos = b;
    }
    out = norm;
  }
  // text[:max_chars] — slice by code points like Python
  std::u32string cps = utf8_decode(out);
  if (static_cast<long long>(cps.size()) > max_chars)
    cps.resize(static_cast<size_t>(max_chars));
  return utf8_encode(cps);
}

ordered_json email_state(const std::string& subject, const std::string& body,
                         const ordered_json& extra, const std::string& sender,
                         bool clean) {
  ordered_json state = ordered_json::object();
  state["subject"] = strip(subject);
  state["body"] = clean ? clean_email_body(body) : body;
  if (!sender.empty()) state["from"] = sender;
  for (auto it = extra.begin(); it != extra.end(); ++it)
    if (!it.value().is_null()) state[it.key()] = it.value();
  return state;
}

}  // namespace snapjudge
