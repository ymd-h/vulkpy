#ifndef VKUTIL_HH
#define VKUTIL_HH

#include <algorithm>
#include <type_traits>
#include <vector>

namespace util {
  template<typename F>
  auto generate_from_range(F&& f, std::uint32_t n) {
    auto v = std::vector<std::invoke_result_t<F, std::uint32_t>>{};
    v.reserve(n);

    auto g = [&f, i=std::uint32_t(0)]() mutable { return f(i++); };
    std::generate_n(std::back_inserter(v), n, g);

    return v;
  }
}

#endif
