// snapjudge-serve entry point.
#include "snapjudge/router.hpp"
#include "snapjudge/serve.hpp"

int main() {
  snapjudge::serve(snapjudge::build_router_from_env());
  return 0;
}
