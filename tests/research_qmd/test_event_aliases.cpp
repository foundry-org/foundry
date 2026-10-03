#include "ResearchEventAliases.h"
#include <cassert>
#include <functional>
#include <iostream>

using namespace foundry;
static void reject(const std::function<void()>& fn) {
  bool rejected = false;
  try { fn(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}
int main() {
  ResearchEventUsage a{{10, {{0, 7}, {3, 6}}}, {20, {{5, 7}}}};
  ResearchEventUsage b{{100, {{0, 7}, {3, 6}}}, {200, {{5, 7}}}};
  const std::set<uintptr_t> ao{10,20}, bo{100,200};
  auto result = research_private_event_aliases(a,b,ao,bo);
  assert(result.at(100)==10 && result.at(200)==20);
  assert(research_private_event_aliases(a,a,ao,ao).at(10)==10);
  assert(research_private_event_aliases({}, {}, {}, {}).empty());
  reject([&]{ research_private_event_aliases(a,b,{10},bo); });
  reject([&]{ research_private_event_aliases(a,b,ao,{100}); });
  reject([&]{ auto v=b;v.at(100).pop_back();research_private_event_aliases(a,v,ao,bo); });
  reject([&]{ auto v=b;v.at(100)[0].second=6;research_private_event_aliases(a,v,ao,bo); });
  reject([&]{ auto v=b;v.at(200)[0].first=7;research_private_event_aliases(a,v,ao,bo); });
  reject([&]{ auto v=b;v.erase(100);research_private_event_aliases(a,v,ao,bo); });
  reject([&]{ auto v=a;v.at(20)=v.at(10);research_private_event_aliases(v,b,ao,bo); });
  reject([&]{ auto v=b;v.at(200)=v.at(100);research_private_event_aliases(a,v,ao,bo); });
  reject([&]{ auto v=b;v.at(100).clear();research_private_event_aliases(a,v,ao,bo); });
  reject([&]{ ResearchEventUsage x{{0,{{0,7}}}};research_private_event_aliases(x,x,{0},{0}); });
  std::cout << "13 private-event alias CPU cases passed\n";
}
