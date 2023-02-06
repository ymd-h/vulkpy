#ifndef VKUTIL_HH
#define VKUTIL_HH

#include <algorithm>
#include <type_traits>
#include <utility>
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

  std::vector<char> readCode(std::string_view name){
    auto f = std::ifstream(name.data(), std::ios::ate | std::ios::binary);
    if(!f.is_open()){
      throw std::runtime_error("failed to open file");
    }
    auto size = f.tellg();
    f.seekg(0);

    auto v = std::vector<char>(size);
    f.read(v.data(), size);

    f.close();
    return v;
  }


  template<typename T, typename F, std::size_t ...I>
  auto pylist2array(F&& f, const pybind11::list& pylist,
                    std::integer_sequence<std::size_t, I...>){
    T array[]{
      pylist[pybind11::size_t(I)].cast<T>()...
    };
    return f(array);
  }
}

#endif
