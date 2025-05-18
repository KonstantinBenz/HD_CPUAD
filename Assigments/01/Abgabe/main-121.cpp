// -------------------------------------------------------------
// main121.cpp  – CPU Algorithm Design · Exercise 1.2.1 including test
// Konstantin Benz 28.04.2025
// g++ -std=c++23 -O2 -Wall -Wextra main121.cpp -o main121.exe
// -------------------------------------------------------------
#include <array>
#include <complex>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <algorithm>
#include <iostream>
#include <iomanip>

// -------------------------------------------------------------
// type aliases from the sheet
// -------------------------------------------------------------
using Bar = std::tuple<
    std::array<std::pair<std::complex<double>, int>, 2>,
    std::pair<std::array<std::complex<float>, 3>, bool>>;

using Foo = std::optional<
    std::variant<
        Bar,
        std::function<Bar(bool)>>>;

// -------------------------------------------------------------
// 1.2.1  accessor functions
// -------------------------------------------------------------
double get_first_double(Foo const &foo)
{
    if (!foo.has_value()) // has_value() is clearer than operator!()
        return 42.0;
    if (!std::holds_alternative<Bar>(*foo))
        return 42.0;

    const Bar &bar = std::get<Bar>(*foo);
    const auto &arrayPairs = std::get<0>(bar);
    const auto &cplx = arrayPairs[0].first;
    return cplx.real();
}

int get_first_int(Foo const &foo)
{
    if (!foo)
        return 42;
    if (!std::holds_alternative<Bar>(*foo))
        return 42;

    const Bar &bar = std::get<Bar>(*foo);
    const auto &arrayPairs = std::get<0>(bar);
    return arrayPairs[0].second;
}

float get_first_float(Foo const &foo)
{
    if (!foo || !std::holds_alternative<Bar>(*foo))
        return 42.0f;

    const Bar &bar = std::get<Bar>(*foo);
    const auto &pair2 = std::get<1>(bar);
    const auto &arrayCplxF32 = pair2.first;
    return arrayCplxF32[0].real();
}

bool get_first_bool(Foo const &foo)
{
    if (!foo || !std::holds_alternative<Bar>(*foo))
        return true;

    const Bar &bar = std::get<Bar>(*foo);
    const auto &p2 = std::get<1>(bar);
    return p2.second;
}

// -------------------------------------------------------------
// utility to print test results
// -------------------------------------------------------------
void print_results(std::string_view label, Foo const &f)
{
    std::cout << label << '\n'
              << "  double : " << get_first_double(f) << '\n'
              << "  int    : " << get_first_int(f) << '\n'
              << "  float  : " << get_first_float(f) << '\n'
              << "  bool   : " << std::boolalpha << get_first_bool(f) << "\n\n";
}

// -------------------------------------------------------------
// test
// -------------------------------------------------------------
int main()
{
    // ---------- case 1: empty Foo  -----------------------------------------
    Foo fooEmpty; // std::nullopt
    print_results("Empty Foo (should be defaults)", fooEmpty);

    // ---------- build a sample Bar ----------------------------------------
    Bar sampleBar{};

    // tuple element 0: array< pair< complex<double>, int >, 2 >
    auto &arrayPairs = std::get<0>(sampleBar);
    arrayPairs[0] = {{1.5, 2.0}, 7};
    arrayPairs[1] = {{3.0, -1.0}, 11};

    // tuple element 1: pair< array< complex<float>,3 >, bool >
    auto &pair2 = std::get<1>(sampleBar);
    auto &arrayCplxFloat = pair2.first;
    arrayCplxFloat[0] = {2.5f, 0.0f};
    arrayCplxFloat[1] = {3.5f, 0.0f};
    arrayCplxFloat[2] = {4.5f, 0.0f};
    pair2.second = false;

    // ---------- case 2: Foo holds Bar -------------------------------------
    Foo fooBar = sampleBar;
    print_results("Foo holding Bar (real values)", fooBar);

    // ---------- case 3: Foo holds std::function<Bar(bool)> -----------------
    Foo fooFunc = std::function<Bar(bool)>(
        [sampleBar](bool)
        { return sampleBar; });
    print_results("Foo holding std::function (defaults again)", fooFunc);

    return 0;
}
