// snapjudge bench: latency of system_one on the CPU forward, C++ side.
// Compare against the Python reference below (bench/bench.py).
#include <chrono>
#include <cstdio>
#include <string>

#include "nlohmann/json.hpp"
#include "snapjudge/agent.hpp"
#include "snapjudge/presets.hpp"

using nlohmann::ordered_json;

int main(int argc, char** argv) {
  int iters = 20;
  std::string dir = argc > 1 ? argv[1] : "";
  if (dir.empty()) dir = "snapjudge/snapjudge";
  snapjudge::Agent agent(dir, "cpu");
  ordered_json state = {{"text", "I was charged twice on my card, please refund the "
                                 "duplicate charge immediately."}};
  ordered_json questions = snapjudge::triage_questions();
  // warmup
  agent.system_one(state, questions);
  double best = 1e30, sum = 0;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    agent.system_one(state, questions);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = ms < best ? ms : best;
    sum += ms;
  }
  std::printf("snapjudge cpu forward: %d iters, mean %.1f ms, best %.1f ms\n",
              iters, sum / iters, best);
  return 0;
}
