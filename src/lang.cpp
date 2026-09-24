#include "snapjudge/lang.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "snapjudge/unicode.hpp"

namespace snapjudge {

// ---------------------------------------------------------------------------
// Script ranges (identical to lang.py _SCRIPT_RANGES)
// ---------------------------------------------------------------------------
static const struct { const char* name; std::vector<std::pair<uint32_t, uint32_t>> ranges; }
    SCRIPT_RANGES[] = {
        {"greek", {{0x0370, 0x03FF}, {0x1F00, 0x1FFF}}},
        {"cyrillic", {{0x0400, 0x052F}, {0x2DE0, 0x2DFF}, {0xA640, 0xA69F}}},
        {"armenian", {{0x0530, 0x058F}}},
        {"hebrew", {{0x0590, 0x05FF}}},
        {"arabic", {{0x0600, 0x06FF}, {0x0750, 0x077F}, {0x08A0, 0x08FF},
                    {0xFB50, 0xFDFF}, {0xFE70, 0xFEFF}}},
        {"devanagari", {{0x0900, 0x097F}, {0xA8E0, 0xA8FF}}},
        {"bengali", {{0x0980, 0x09FF}}},
        {"gurmukhi", {{0x0A00, 0x0A7F}}},
        {"gujarati", {{0x0A80, 0x0AFF}}},
        {"oriya", {{0x0B00, 0x0B7F}}},
        {"tamil", {{0x0B80, 0x0BFF}}},
        {"telugu", {{0x0C00, 0x0C7F}}},
        {"kannada", {{0x0C80, 0x0CFF}}},
        {"malayalam", {{0x0D00, 0x0D7F}}},
        {"sinhala", {{0x0D80, 0x0DFF}}},
        {"thai", {{0x0E00, 0x0E7F}}},
        {"lao", {{0x0E80, 0x0EFF}}},
        {"tibetan", {{0x0F00, 0x0FFF}}},
        {"myanmar", {{0x1000, 0x109F}}},
        {"georgian", {{0x10A0, 0x10FF}}},
        {"ethiopic", {{0x1200, 0x137F}}},
        {"khmer", {{0x1780, 0x17FF}}},
        {"hangul", {{0x1100, 0x11FF}, {0x3130, 0x318F}, {0xAC00, 0xD7AF}}},
        {"kana", {{0x3040, 0x309F}, {0x30A0, 0x30FF}, {0x31F0, 0x31FF}}},
        {"han", {{0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}}},
};

static inline bool is_latin_fast(char32_t cp) {
  return cp < 0x02B0 || (cp >= 0x1E00 && cp <= 0x1EFF) ||
         (cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A);
}

// ---------------------------------------------------------------------------
// Stopword lists — verbatim from lang.py _STOP
// ---------------------------------------------------------------------------
using Words = std::unordered_set<std::string>;
static const std::unordered_map<std::string, Words>& stop_lists() {
  static const std::unordered_map<std::string, Words> lists = {
      {"en", {"the","and","is","are","was","were","to","of","in","for","with","that",
              "this","it","you","have","has","not","but","on","at","be","as","from",
              "will","can","would","there","their","what","which","please","we","i"}},
      {"fr", {"le","la","les","des","une","est","pour","dans","que","qui","avec","sur",
              "pas","plus","nous","vous","être","cette","mais","sont","ont","aux","ce",
              "et","du","au","ou","je","tu","il","elle","ils","elles","mon","ton",
              "ma","ta","sa","mes","tes","ses","ces","deux","trois","très","bien",
              "tout","tous","toute","fait","veux","veut","peux","peut","dois","doit",
              "merci","bonjour","jour","jours","mois","fois","quand","comment","pourquoi",
              "alors","donc"}},
      {"de", {"der","die","das","und","ist","ein","eine","den","dem","nicht","mit","für",
              "auf","von","zu","sich","auch","werden","wurde","haben","sind","oder","aber"}},
      {"es", {"el","los","las","que","por","con","para","una","es","se","del","como",
              "pero","son","está","este","esta","todo","más","muy","hay","sus",
              "la","un","y","al","lo","le","les","su","mi","tu","nos",
              "ni","dos","tres","fue","fueron","ser","tiene","tienen","tengo","puede",
              "pueden","quiero","necesito","hemos","han","sobre","entre","cuando","donde",
              "porque","aunque","también","ya","eso","esto","esa","ese","nada","algo",
              "aquí","hoy","gracias"}},
      {"pt", {"os","as","que","em","um","uma","para","com","não","é","se","do","da",
              "dos","das","mas","são","está","este","esta","muito","pelo","pela",
              "o","e","na","nas","nos","ao","aos","por","foi","era","ser","sou",
              "tem","tenho","pode","podem","quero","preciso","eu","meu","minha","seu",
              "sua","isso","isto","aqui","ali","como","quando","onde","porque","mais",
              "já","ainda","agora","hoje","ontem","dois","três","tudo","nada","obrigado",
              "olá",
              "você","vocês","voce","voces","vc","vcs","nao","sao","ja","até","tá","pra",
              "gostaria","obrigada","também","tambem","estou","estamos","meus","minhas",
              "nosso","nossa","consigo","cadê","boa","tarde","noite",
              "depois","antes","então","entao","ninguém","ninguem","alguém","alguem","nenhum",
              "nenhuma","estava","ficou","fiz","deu"}},
      {"it", {"il","lo","gli","che","di","per","con","non","è","si","del","della","sono",
              "questo","questa","anche","come","più","sono","nella","alla",
              "la","le","un","uno","una","e","ed","o","da","su","tra","fra","mi",
              "ci","ne","ho","hai","ha","abbiamo","avete","hanno","era","stato","stata",
              "devo","deve","devono","voglio","vorrei","mio","mia","tuo","sua","quando",
              "dove","perche","molto","poco","sempre","mai","già","ancora","adesso","oggi",
              "ieri","grazie","ciao","scusa",
              "nel","nell","negli","sul","sulla","sulle","dal","dalla","dallo","dagli","dei",
              "delle","dello","degli","agli","alle","col"}},
      {"nl", {"het","een","van","is","op","te","dat","niet","met","voor","zijn","aan",
              "door","maar","ook","worden","deze","naar","wordt"}},
      {"ro", {"și","să","este","sunt","care","pentru","din","dar","după","până","fără",
              "ale","lui","în","fost","acum","vreau","trebuie","foarte","acest","această",
              "acesta","aceasta","mi","ți","vă","nu"}},
      {"bn", {"ami","amar","amake","amra","amader","apni","apnar","apnake","apnara",
              "tumi","tomar","tomake","tomra","tader","ota","eita","oita",
              "ekta","ei","oi","ki","keno","kivabe","kibhabe","kothay","kokhon","kobe",
              "koto","kintu","jodi","tahole","ar","theke","jonno","sathe","shathe","diye",
              "niye","moddhe","kore","korte","korchi","korsi","korbo","korechi","koreche",
              "korun","koren","korlam","hobe","hoyeche","hoise","hocche","hoyni",
              "chai","chaina","lagbe","parchi","parbo","parchina","peyechi","paini",
              "dite","dilam","diyechi","nai","khub","onek","ekhon","akhon","ekhono",
              "abar","ekbar","duibar","ajke","kalke","taka","bhalo","valo","kharap",
              "shomossa","somossa","dhonnobad","bhai","shob","keu","kichu","bolte","bolun",
              "parben","asbe","jabe","pabo","ferot","dorkar","hoye","geche","gese"}},
      {"az", {"və","ve","bir","bu","üçün","ucun","ilə","ile","olan","olub","olmasa",
              "var","yox","yoxdur","mən","sən","biz","siz","onlar","daha","çox","cox",
              "hər","nə","kimi","görə","sonra","əgər","eger","deyil","lakin","amma",
              "ancaq","artıq","artiq","də","isə","həm","yalnız","yalniz"}},
  };
  return lists;
}

// Words claimed by more than one list.
static const Words& shared_words() {
  static const Words shared = [] {
    std::unordered_map<std::string, int> counts;
    for (const auto& [lg, ws] : stop_lists())
      for (const auto& w : ws) ++counts[w];
    Words out;
    for (const auto& [w, n] : counts)
      if (n > 1) out.insert(w);
    return out;
  }();
  return shared;
}

// ---------------------------------------------------------------------------
// Non-English diacritic set (unicode chars stored in a utf-8 set)
// ---------------------------------------------------------------------------
static const Words& non_en_diacritics() {
  static const Words d = [] {
    std::u32string cps =
        U"àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ"
        U"ăâîșțşţ"
        U"ąćęłńśźż"
        U"čďěňřšťůž"
        U"őű"
        U"ğı"
        U"āēģīķļņūž"
        U"đ"
        U"ə";
    Words out;
    for (char32_t cp : cps) out.insert(utf8_encode(cp));
    return out;
  }();
  return d;
}

// ---------------------------------------------------------------------------
// state flattening
// ---------------------------------------------------------------------------
static void iter_text(const ordered_json& v, int depth, std::vector<std::string>& out) {
  if (depth > 6 || v.is_null()) return;
  if (v.is_string()) {
    out.push_back(v.get<std::string>());
  } else if (v.is_object()) {
    for (auto it = v.begin(); it != v.end(); ++it) iter_text(it.value(), depth + 1, out);
  } else if (v.is_array()) {
    for (const auto& e : v) iter_text(e, depth + 1, out);
  }
}

std::string state_text(const ordered_json& state, int max_chars) {
  std::vector<std::string> parts;
  iter_text(state, 0, parts);
  std::string out;
  bool first = true;
  for (const auto& p : parts) {
    if (!first) out += " ";
    out += p;
    first = false;
  }
  // Python slices by chars; our strings are UTF-8 — slice by code points.
  std::u32string cps = utf8_decode(out);
  if (static_cast<long long>(cps.size()) > max_chars) cps.resize(static_cast<size_t>(max_chars));
  return utf8_encode(cps);
}

// ---------------------------------------------------------------------------
// script detection
// ---------------------------------------------------------------------------
namespace {
// Insertion-ordered profile: first "latin", then scripts in scan-appearance
// order — exactly how the Python dict is built (counts={"latin":0} first, then
// increments in scan order), which matters for max() tie-breaking.
std::vector<std::pair<std::string, double>> script_profile_ordered(const std::string& text) {
  std::vector<std::pair<std::string, int>> counts{{"latin", 0}};
  auto bump = [&](const std::string& nm) {
    for (auto& [k, v] : counts)
      if (k == nm) { ++v; return; }
    counts.emplace_back(nm, 1);
  };
  for (char32_t cp : utf8_decode(text)) {
    if (!uni_isalpha(cp)) continue;
    if (is_latin_fast(cp)) { bump("latin"); continue; }
    std::string nm;
    for (const auto& sr : SCRIPT_RANGES) {
      bool hit = false;
      for (auto [lo, hi] : sr.ranges)
        if (static_cast<uint32_t>(cp) >= lo && static_cast<uint32_t>(cp) <= hi) {
          nm = sr.name;
          hit = true;
          break;
        }
      if (hit) break;
    }
    if (!nm.empty()) bump(nm);
    else bump("other");
  }
  int total = 0;
  for (const auto& [k, v] : counts) total += v;
  if (!total) return {};
  std::vector<std::pair<std::string, double>> out;
  for (const auto& [k, v] : counts)
    if (v) out.emplace_back(k, static_cast<double>(v) / total);
  return out;
}
}  // namespace

std::unordered_map<std::string, double> script_profile(const std::string& text) {
  std::unordered_map<std::string, double> out;
  for (auto& [k, v] : script_profile_ordered(text)) out[k] = v;
  return out;
}

std::string detect_script(const std::string& text) {
  // Python dicts build in insertion order and counts land at the position of
  // first insertion; "latin" is added AT THE END (counts["latin"] = latin).
  // max(counts.items(), key=count) returns the first max in that order.
  std::vector<std::string> order;
  std::unordered_map<std::string, int> counts;
  int latin = 0;
  auto bump = [&](const std::string& nm, int by) {
    if (!counts.count(nm)) order.push_back(nm);
    counts[nm] += by;
  };
  for (char32_t cp : utf8_decode(text)) {
    if (!uni_isalpha(cp)) continue;
    if (is_latin_fast(cp)) { ++latin; continue; }
    std::string nm = "other";
    for (const auto& sr : SCRIPT_RANGES) {
      bool hit = false;
      for (auto [lo, hi] : sr.ranges)
        if (static_cast<uint32_t>(cp) >= lo && static_cast<uint32_t>(cp) <= hi) {
          nm = sr.name;
          hit = true;
          break;
        }
      if (hit) break;
    }
    bump(nm, 1);
  }
  bump("latin", latin);
  int total = 0;
  for (const auto& nm : order) total += counts[nm];
  if (total == 0) return "unknown";
  std::string best;
  int bn = -1;
  for (const auto& nm : order)
    if (counts[nm] > bn) { bn = counts[nm]; best = nm; }
  return best;
}

// ---------------------------------------------------------------------------
// latin profile
// ---------------------------------------------------------------------------
static constexpr double NON_EN_DIACRITIC_RATE = 0.02;
static constexpr double NON_LATIN_FRACTION = 0.2;
static constexpr double NON_LATIN_MIN_FRACTION = 0.1;
static constexpr int NON_LATIN_MIN_LETTERS = 10;

// _WORD = [^\W\d_]+ over unicode; _IDENTIFIER = [\w-]*([.@][\w-]+)+ replaced by " "
static std::string strip_identifiers(const std::u32string& cps) {
  // An identifier stretches [wordchar|'-']*('.'|'@')(wordchar...)+ possibly
  // repeated. Implement as a scan: for maximal runs over [\w\-.@]+, if the run
  // contains '.' or '@' with \w on both sides (per the regex), drop it.
  std::u32string out;
  size_t i = 0, n = cps.size();
  auto is_w = [](char32_t c) { return uni_iswordchar(c) || c == U'-'; };
  while (i < n) {
    // try to match identifier at i: [\w-]*([.@][\w-]+)+
    size_t j = i;
    while (j < n && is_w(cps[j])) ++j;
    size_t k = j;
    size_t last = j;
    while (k < n && (cps[k] == U'.' || cps[k] == U'@')) {
      size_t m = k + 1;
      while (m < n && is_w(cps[m])) ++m;
      if (m == k + 1) break;  // need at least one \w- after the separator
      last = m;
      k = m;
    }
    if (last > j) {
      out.push_back(U' ');
      i = last;
    } else {
      out.push_back(cps[i]);
      ++i;
    }
  }
  return utf8_encode(out);
}

static std::vector<std::string> word_scan(const std::u32string& cps) {
  std::vector<std::string> out;
  std::u32string cur;
  for (char32_t cp : cps) {
    if (uni_isword(cp)) cur.push_back(cp);
    else if (!cur.empty()) { out.push_back(utf8_encode(cur)); cur.clear(); }
  }
  if (!cur.empty()) out.push_back(utf8_encode(cur));
  return out;
}

LatinProfile latin_profile(const std::string& text) {
  // 'İ'.lower() has a combining dot; Python handles it via replace before lower.
  std::string stripped = strip_identifiers(utf8_decode(text));
  std::string norm = stripped;
  {
    // text.replace("İ","i")
    size_t pos = 0;
    const std::string from = "İ";
    while ((pos = norm.find(from, pos)) != std::string::npos) {
      norm.replace(pos, from.size(), "i");
      pos += 1;
    }
  }
  std::u32string lowered_cps = uni_lower(utf8_decode(norm));
  std::vector<std::string> words = word_scan(lowered_cps);

  std::string lowered = utf8_encode(uni_lower(utf8_decode(text)));
  long diac = 0;
  for (char32_t cp : utf8_decode(lowered)) {
    if (non_en_diacritics().count(utf8_encode(cp))) ++diac;
  }
  double diac_rate = static_cast<double>(diac) /
                     std::max<size_t>(1, utf8_decode(lowered).size());
  bool non_english = diac_rate >= NON_EN_DIACRITIC_RATE;
  LatinProfile prof;
  prof.diacritic_rate = diac_rate;
  prof.looks_non_english = non_english;
  if (words.size() < 4) return prof;

  std::unordered_set<std::string> word_set(words.begin(), words.end());
  std::unordered_map<std::string, int> scores;
  for (const auto& [lg, ws] : stop_lists()) {
    int s = 0;
    for (const auto& w : words) s += ws.count(w) ? 1 : 0;
    scores[lg] = s;
  }
  int en = scores["en"];
  prof.english_hits = en;

  std::string best_lg;
  int best = 0;
  // evidenced: non-en languages with at least one own (non-shared) word hit
  for (const auto& [lg, ws] : stop_lists()) {
    if (lg == "en") continue;
    bool evidenced = false;
    for (const auto& w : word_set)
      if (ws.count(w) && !shared_words().count(w)) { evidenced = true; break; }
    if (!evidenced) continue;
    if (scores[lg] > best || (scores[lg] == best && !best_lg.empty())) {
      // Python max() keeps the first max in dict order
      if (scores[lg] > best) { best = scores[lg]; best_lg = lg; }
    }
  }

  if (!best_lg.empty() && best >= std::max(2, en + 2)) {
    prof.language = best_lg;
  } else if (!best_lg.empty() && non_english && best >= std::max(2, en)) {
    prof.language = best_lg;
  } else if (en && !non_english) {
    prof.language = "en";
  }
  return prof;
}

std::string guess_latin_language(const std::string& text) {
  return latin_profile(text).language;
}

// ---------------------------------------------------------------------------
// _non_latin_words
// ---------------------------------------------------------------------------
static std::string script_of(char32_t cp) {
  if (cp < 0x0250 || (cp >= 0x1E00 && cp <= 0x1EFF) ||
      (cp >= 0xFF21 && cp <= 0xFF3A) || (cp >= 0xFF41 && cp <= 0xFF5A))
    return "";
  for (const auto& sr : SCRIPT_RANGES)
    for (auto [lo, hi] : sr.ranges)
      if (static_cast<uint32_t>(cp) >= lo && static_cast<uint32_t>(cp) <= hi)
        return sr.name;
  return "";
}

static std::vector<std::u32string> non_latin_words(const std::string& text) {
  std::vector<std::u32string> runs;
  std::u32string cur;
  std::string script;
  for (char32_t cp : utf8_decode(text)) {
    if (uni_iscombining(cp)) continue;
    std::string s = script_of(cp);
    if (!s.empty() && s == script) {
      cur.push_back(cp);
      continue;
    }
    if (!cur.empty()) runs.push_back(cur);
    if (!s.empty()) {
      cur = std::u32string(1, cp);
      script = s;
    } else {
      cur.clear();
      script.clear();
    }
  }
  if (!cur.empty()) runs.push_back(cur);
  std::vector<std::u32string> out;
  for (const auto& w : runs) {
    if (w.size() >= 2 && !uni_isupper(w[0])) out.push_back(w);
  }
  return out;
}

// ---------------------------------------------------------------------------
// analyse
// ---------------------------------------------------------------------------
ordered_json analyse(const ordered_json& state) {
  std::string text = state_text(state);
  auto prof = script_profile_ordered(text);
  std::string script = detect_script(text);
  double non_latin = 0.0;
  if (!prof.empty()) {
    double latin_share = 0.0;
    for (const auto& [k, v] : prof)
      if (k == "latin") latin_share = v;
    non_latin = std::round((1.0 - latin_share) * 10000) / 10000;
  }
  long letters = 0;
  for (char32_t cp : utf8_decode(text)) letters += uni_isalpha(cp) ? 1 : 0;
  long n_non_latin = std::lround(non_latin * letters);

  if (script == "latin" && !non_latin_words(text).empty() &&
      (non_latin >= NON_LATIN_FRACTION ||
       (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS))) {
    // Python: max((s for s in prof if s != "latin"), key=prof.get) — first max
    // in insertion order wins.
    std::string best;
    double bestv = -1;
    for (const auto& [k, v] : prof)
      if (k != "latin" && v > bestv) { bestv = v; best = k; }
    if (!best.empty()) script = best;
  }

  ordered_json sp = ordered_json::object();
  for (const auto& [k, v] : prof) sp[k] = v;

  if (script == "unknown") {
    return ordered_json{{"script", "unknown"},         {"script_profile", sp},
                        {"language", nullptr},          {"is_english", true},
                        {"language_undecided", true},   {"diacritic_rate", 0.0},
                        {"non_latin_fraction", 0.0}};
  }
  if (script != "latin") {
    return ordered_json{{"script", script},           {"script_profile", sp},
                        {"language", nullptr},        {"is_english", false},
                        {"language_undecided", true}, {"diacritic_rate", 0.0},
                        {"non_latin_fraction", non_latin}};
  }
  LatinProfile lp = latin_profile(text);
  std::string lang = lp.language;
  bool undecided = lang.empty();
  bool english = (lang == "en") || (undecided && !lp.looks_non_english);
  return ordered_json{{"script", "latin"},
                      {"script_profile", sp},
                      {"language", undecided ? ordered_json(nullptr) : ordered_json(lang)},
                      {"is_english", english},
                      {"language_undecided", undecided},
                      {"diacritic_rate", std::round(lp.diacritic_rate * 10000.0) / 10000.0},
                      {"non_latin_fraction", non_latin}};
}

bool is_english(const ordered_json& state) {
  return analyse(state)["is_english"].get<bool>();
}

}  // namespace snapjudge
