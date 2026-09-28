#include "snapjudge/email.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "snapjudge/unicode.hpp"

#ifdef SNAPJUDGE_HAVE_PCRE2
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#endif

namespace snapjudge {

namespace {

// Thin PCRE2 wrapper. Compiles once with UTF+UCP (the Python `re.UNICODE`
// equivalent); `search` matches anywhere, `match_span` returns the group-0 span.
// The code handle is shared so `Re` is safely copyable into the static tables.
//
// When PCRE2 is unavailable (SNAPJUDGE_HAVE_PCRE2 unset — the WASM / no-deps
// build), `Re` is a no-op that never matches: clean_email_body still runs, but
// skips the regex-based history/signature/disclaimer stripping.
#ifdef SNAPJUDGE_HAVE_PCRE2
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
#else
struct Re {
  Re(const char*, bool = false) {}
  bool search(const std::string&) const { return false; }
  std::pair<size_t, size_t> match_span(const std::string&, size_t = 0) const {
    return {std::string::npos, std::string::npos};
  }
};
#endif

// ---------------------------------------------------------------------------
// Quoted-history markers. A reply header ("On Tue, ... wrote:") or a forwarded-
// message separator ends the new message: everything from that line down is
// quoted history.
// ---------------------------------------------------------------------------
const std::vector<Re>& quote_headers() {
  static const std::vector<Re> v = [] {
    std::vector<Re> t;
    t.emplace_back(R"(^\s*On .{0,300}wrote:\s*$)", true);
    t.emplace_back(R"(^\s*Em (?=.*\d).{0,300}escreveu:\s*$)", true);
    t.emplace_back(R"(^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$)", true);
    t.emplace_back(R"(^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,})", true);
    t.emplace_back(R"(^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,})", true);
    t.emplace_back(R"(^\s*_{8,}\s*$)");
    // `From:`/`De:` open ordinary prose too ("From: my side the integration
    // works"), so a header is only recognised when an address (an @ or <)
    // follows, which prose never has.
    t.emplace_back(R"(^\s*From:\s.*[@<])", true);
    t.emplace_back(R"(^\s*De:\s.*[@<])", true);
    return t;
  }();
  return v;
}

// Gmail wraps a long attribution line, leaving "user@x.com> wrote:" alone on
// the next line. The tail cuts too, and takes the "On/Em/El ..." head it
// belongs to with it.
const Re& attribution_tail() {
  static const Re r(R"(^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$)", true);
  return r;
}
const Re& attribution_head() {
  static const Re r(R"(^\s*(On|Em|El) (?=.*\d))", true);
  return r;
}

// Exchange often leaves the address out of the reply header ("De: Maria
// Souza"), so a bare `De:`/`From:` name line only counts as a header when the
// next line is the header's own `Sent:`/`Date:` (or the Portuguese/Spanish
// equivalents).
const Re& header_from_name() {
  static const Re r(R"(^\s*(De|From):\s+\S)", true);
  return r;
}
const Re& header_next() {
  static const Re r(R"(^\s*(Enviad[oa]( em| el)?:\s|Sent:\s|(Data|Fecha|Date):\s.*\d{4}))", true);
  return r;
}

// ---------------------------------------------------------------------------
// Signature markers. A closing line is the closing word plus punctuation and
// at most a name; anything more is a sentence. The closing word is matched
// case-insensitively, the name is not (a name is capitalised, "for" in "Thanks
// for the quick reply." is not).
// ---------------------------------------------------------------------------
const std::vector<Re>& signature_markers() {
  static const std::vector<Re> v = [] {
    std::vector<Re> t;
    t.emplace_back(R"(^\s*--\s*$)");
    t.emplace_back(
        R"(^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?[\s,;:!.]*(?:[^\W\d_a-z\xDF-\xF6\xF8-\xFF][\w'-]*[\s,.]*){0,3}$)");
    t.emplace_back(R"(^\s*sent from my (iphone|android|mobile|ipad))", true);
    // Portuguese/Spanish closings match only on their own line — "Obrigado pelo
    // retorno, mas ..." is a request, not a signature — so no trailing words.
    t.emplace_back(
        R"(^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$)",
        true);
    return t;
  }();
  return v;
}

// Mobile and mail-app footers. Only a line that is nothing *but* the footer
// matches — "Sent from my phone, the attachment is attached." is a request.
const Re& device_footer() {
  static const Re r(
      R"(^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?) (iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows)( (iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows|para|for|no|na|\d+))*|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$)",
      true);
  return r;
}

// Confidentiality disclaimers (English, Portuguese, Spanish). Tied to the noun
// they protect so a request that merely *mentions* "confidential" ("Is this
// confidential?") is not deleted whole.
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
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
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
  std::string t = lstrip(s);
  return !t.empty() && t[0] == '>';
}

// Split after a sentence terminator (keeping the terminator with its sentence).
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
      return !(lo.size() == 1 && lo[0] == cp);  // isupper: lowering changed it
    }
  }
  return false;
}

// An unpunctuated request line glued to a disclaimer line ("locked\nThis ...")
// splits at the newline because the next line starts uppercase; a lowercase
// continuation ("are\nconfidential") belongs to the same sentence.
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

// Drop boilerplate disclaimer text from one paragraph. A paragraph is dropped
// whole only when every sentence in it is boilerplate; otherwise only the
// boilerplate sentences go.
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

  // Bound regex work before the expensive patterns below: the disclaimer
  // regex uses [^.]{0,60/80/100} alternations whose cost grows with input
  // length, and only max_chars are ever returned. Truncate early so one
  // megabyte-long line cannot dominate matching.
  if (text.size() > static_cast<size_t>(max_chars) * 4)
    text.resize(static_cast<size_t>(max_chars) * 4);

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

  // Cut at the first signature/device footer found in the lower 40% of lines
  // (signatures live near the end).
  size_t cut = lines.size();
  size_t scan_from = std::max<size_t>(1, std::min(static_cast<size_t>(lines.size() * 0.6),
                                                  lines.size() >= 8 ? lines.size() - 8 : 0));
  for (size_t i = scan_from; i < lines.size(); ++i) {
    std::string t = strip(lines[i]);
    size_t nch = utf8_decode(t).size();
    if ((nch <= 40 && std::any_of(signature_markers().begin(), signature_markers().end(),
                                  [&](const Re& re) { return re.search(lines[i]); })) ||
        (nch <= 60 && device_footer().search(lines[i]))) {
      cut = i;
      break;
    }
  }
  lines.resize(cut);

  // Split on blank lines into paragraphs, then drop disclaimers per paragraph.
  std::string joined = join(lines, "\n");
  std::vector<std::string> paragraphs;
  {
    static const Re para_re(R"(\n\s*\n)");
    size_t pos = 0;
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

  // Collapse runs of spaces/tabs into one space.
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

  // Slice to max_chars code points (like Python slicing a string).
  std::u32string cps = utf8_decode(out);
  if (static_cast<long long>(cps.size()) > max_chars) cps.resize(static_cast<size_t>(max_chars));
  return utf8_encode(cps);
}

ordered_json email_state(const std::string& subject, const std::string& body,
                         const ordered_json& extra, const std::string& sender, bool clean) {
  ordered_json state = ordered_json::object();
  state["subject"] = strip(subject);
  state["body"] = clean ? clean_email_body(body) : body;
  if (!sender.empty()) state["from"] = sender;
  for (auto it = extra.begin(); it != extra.end(); ++it)
    if (!it.value().is_null()) state[it.key()] = it.value();
  return state;
}

}  // namespace snapjudge
