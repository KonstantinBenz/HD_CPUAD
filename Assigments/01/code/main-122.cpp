// -------------------------------------------------------------
// main.cpp  – CPU Algorithm Design · Exercise 1.2.2 including test
// Konstantin Benz 28.04.2025
// g++ -std=c++23 -O2 -Wall -Wextra main122.cpp -o main122.exe
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
// 1.2.2  –  How Does This std::function?
// -------------------------------------------------------------
Bar get_Bar(Foo const &foo)
{
    // (1)  optional is empty  ->  default-constructed Bar
    if (!foo)
        return Bar{};

    // (2)  inspect the variant inside
    const auto &variant = *foo;

    // (2a) holds a real Bar
    if (std::holds_alternative<Bar>(variant))
        return std::get<Bar>(variant);

    // (2b) otherwise it must be the std::function<Bar(bool)>
    const auto &fn = std::get<std::function<Bar(bool)>>(variant);
    return fn(true); // call with argument true
}

// -------------------------------------------------------------
// utility to build a non-trivial Bar to see real numbers in the output
// -------------------------------------------------------------
Bar make_sample_bar()
{
    Bar b{};

    // tuple element 0  (array< pair< complex<double>, int >, 2 >)
    auto &arrPairs = std::get<0>(b);
    arrPairs[0] = {{1.2, 0.0}, 7};
    arrPairs[1] = {{3.4, 0.0}, 11};

    // tuple element 1  (pair< array< complex<float>,3 >, bool >)
    auto &pair2 = std::get<1>(b);
    auto &arrCplxFloat = pair2.first;
    arrCplxFloat[0] = {2.5f, 0.0f};
    pair2.second = false;

    return b;
}

// -------------------------------------------------------------
// test
// -------------------------------------------------------------
int main()
{
    Bar sample = make_sample_bar();

    // --- Case A: empty Foo ------------------------------------
    Foo fooEmpty;
    Bar a = get_Bar(fooEmpty);
    std::cout << "empty -> first int = "
              << std::get<0>(a)[0].second // default-constructed Bar -> 0
              << '\n';

    // --- Case B: Foo holding Bar ------------------------------
    Foo fooBar = sample;
    Bar b = get_Bar(fooBar);
    std::cout << "bar   -> first int = "
              << std::get<0>(b)[0].second // 7
              << '\n';

    // --- Case C: Foo holding std::function<Bar(bool)> ---------
    Foo fooFunc = std::function<Bar(bool)>(
        [sample](bool)
        { return sample; });
    Bar c = get_Bar(fooFunc);
    std::cout << "func  -> first int = "
              << std::get<0>(c)[0].second // 7
              << '\n';
}