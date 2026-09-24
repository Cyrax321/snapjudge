// snapjudge CLI — port of cli.py.
//
//   snapjudge "I was charged twice"                 routing only (offline)
//   snapjudge "I was charged twice" --predict       full answers
//   snapjudge                                       interactive mode
//   snapjudge "Mein Konto wurde zweimal belastet" --lang de

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "snapjudge/presets.hpp"
#include "snapjudge/router.hpp"

using nlohmann::ordered_json;
using snapjudge::Router;

namespace {

struct Args {
  std::vector<std::string> text;
  bool predict = false;
  std::string model, lang, task, device;
  bool json_out = false;
  bool help = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string t = argv[i];
    auto take = [&](std::string& dst) {
      if (i + 1 < argc) dst = argv[++i];
    };
    if (t == "--predict") a.predict = true;
    else if (t == "--json") a.json_out = true;
    else if (t == "--model") take(a.model);
    else if (t == "--lang") take(a.lang);
    else if (t == "--task") take(a.task);
    else if (t == "--device") take(a.device);
    else if (t == "--help" || t == "-h") a.help = true;
    else a.text.push_back(t);
  }
  return a;
}

void print_help() {
  std::puts(
      "snapjudge — test snapjudge locally: route or answer a request from the\n"
      "command line.\n"
      "usage: snapjudge [text...] [--predict] [--model M] [--lang L] [--task T]\n"
      "                 [--device D] [--json]\n");
}

void show_decision(const ordered_json& d) {
  std::cout << "Model     : " << d["model"].get<std::string>() << "\n"
            << "Reason    : " << d["reason"].get<std::string>() << "\n";
  if (!d.value("detection", ordered_json()).is_null())
    std::cout << "Detected  : " << d["detection"].dump(-1, ' ', false) << "\n";
}

void show_answers(const ordered_json& result) {
  if (!result.value("routing", ordered_json()).is_null()) {
    show_decision(result["routing"]);
    std::cout << "\n";
  }
  for (auto it = result["answers"].begin(); it != result["answers"].end(); ++it) {
    const ordered_json& a = it.value();
    std::ostringstream detail;
    if (a.contains("choice"))
      detail << a["choice"].get<std::string>();
    else if (a.contains("score"))
      detail << a["score"].get<double>();
    else if (a.contains("noul"))
      detail << a["noul"].get<double>();
    else
      detail << a.dump(-1, ' ', false);
    char pad[64];
    std::snprintf(pad, sizeof(pad), "%-12.12s", it.key().c_str());
    std::cout << pad << ": " << detail.str() << "\n";
  }
}

int run(const std::string& text, const Args& a, Router* router) {
  Router::Options o;
  o.device = a.device;
  o.preload = false;
  auto local = [o]() { return new Router(o); };
  Router* r = router ? router : local();
  try {
    if (a.predict) {
      ordered_json result = r->predict(ordered_json{{"text", text}},
                                       snapjudge::router_questions(),
                                       a.model, a.task, a.lang);
      if (a.json_out)
        std::cout << result.dump(2, ' ', false) << "\n";
      else
        show_answers(result);
    } else {
      ordered_json d = r->route(ordered_json{{"text", text}},
                                ordered_json::object(), a.model, a.task, a.lang);
      if (a.json_out)
        std::cout << d.dump(2, ' ', false) << "\n";
      else
        show_decision(d);
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "snapjudge: " << e.what() << "\n";
    if (!router) delete r;
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "snapjudge: could not run snapjudge (" << e.what() << ").\n"
                 "Check that the checkpoints are reachable (HF cache or network on "
                 "first use).\n";
    if (!router) delete r;
    return 2;
  }
  if (!router) delete r;
  return 0;
}

int interactive(const Args& a) {
  Router::Options o;
  o.device = a.device;
  Router router(o);
  std::cout << "snapjudge interactive mode. Type a request and press Enter; Ctrl-D or "
               "'quit' to exit.\n";
  std::string line;
  while (true) {
    std::cout << "snapjudge> " << std::flush;
    if (!std::getline(std::cin, line)) {
      std::cout << "\n";
      break;
    }
    size_t a0 = line.find_first_not_of(" \t");
    if (a0 == std::string::npos) continue;
    line = line.substr(a0);
    std::string low = line;
    for (auto& c : low) c = std::tolower(static_cast<unsigned char>(c));
    if (low == "quit" || low == "exit") break;
    run(line, a, &router);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.help) {
    print_help();
    return 0;
  }
  std::string text;
  for (size_t i = 0; i < a.text.size(); ++i) {
    if (i) text += " ";
    text += a.text[i];
  }
  // strip
  size_t first = text.find_first_not_of(" \t");
  if (first == std::string::npos) return interactive(a);
  return run(text.substr(first), a, nullptr);
}
