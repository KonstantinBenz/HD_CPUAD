// -------------------------------------------------------------
// main.cpp  – CPU Algorithm Design · Exercise 1.2.3 including test
// Konstantin Benz 28.04.2025
// g++ -std=c++23 -O2 -Wall -Wextra main123.cpp -o main123.exe
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
// 1.2.3 – element-wise multiplication of two Bar
// -------------------------------------------------------------
Bar multiply(Bar const& x, Bar const& y) {
    // 1) Structured-bind the top-level tuple into (array of pairs, pair<array,bool>)
    auto [arrPairsX, pair2X] = x;
    auto [arrPairsY, pair2Y] = y;

    // 2) Multiply the first element: array< pair<complex<double>,int>, 2 >
    //    resultArrayPairs[i] = { arrPairsX[i].first * arrPairsY[i].first,
    //                            arrPairsX[i].second * arrPairsY[i].second }
    std::array<std::pair<std::complex<double>,int>,2> resultArrayPairs;
    std::transform(
        arrPairsX.begin(), arrPairsX.end(),   // first input range
        arrPairsY.begin(),                    // second input range
        resultArrayPairs.begin(),             // where to store results
        [](auto const& u, auto const& v) {    // lambda multiplies pair elements
            return std::pair{
                u.first  * v.first,   // complex<double> * complex<double>
                u.second * v.second   // int * int
            };
        }
    );

    // 3) Multiply the second element’s array of complex<float>
    //    resultArrayCplx[i] = pair2X.first[i] * pair2Y.first[i]
    std::array<std::complex<float>,3> resultArrayCplx;
    std::transform(
        pair2X.first.begin(), pair2X.first.end(),
        pair2Y.first.begin(),
        resultArrayCplx.begin(),
        [](auto const& a, auto const& b) {
            return a * b;   // complex<float> * complex<float>
        }
    );

    // 4) “Multiply” the two bools by logical AND
    bool resultBool = pair2X.second && pair2Y.second;

    // 5) Pack everything back into a Bar and return
    return Bar{ resultArrayPairs,
                std::pair{ resultArrayCplx, resultBool } };
}

// -------------------------------------------------------------
// utility to to create a sample Bar with seed s
// -------------------------------------------------------------
Bar make_bar(double s) {
    Bar b{};
    auto& ap = std::get<0>(b);
    ap[0] = { {s, 0.0}, int(s) };
    ap[1] = { {s+1, 0.0}, int(s)+1 };
    auto& p2 = std::get<1>(b);
    auto& arrF = p2.first;
    arrF[0] = {float(s), 0.0f};
    arrF[1] = {float(s)+1, 0.0f};
    arrF[2] = {float(s)+2, 0.0f};
    p2.second = (int(s) % 2 == 0);
    return b;
}

// -------------------------------------------------------------
// test
// -------------------------------------------------------------
int main() {
    // Prepare two Bars
    Bar b1 = make_bar(2.0); // ints: 2,3  bool: true
    Bar b2 = make_bar(3.0); // ints: 3,4  bool: false

    // Multiply
    Bar prod = multiply(b1, b2);

    // Extract results
    auto& ap  = std::get<0>(prod);
    auto& p2  = std::get<1>(prod);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "complex<double>[0].real = " << ap[0].first.real() << "\n"; // 6.0
    std::cout << "int[0]                = " << ap[0].second        << "\n"; // 6
    std::cout << "complex<float>[0].real = " << p2.first[0].real()  << "\n"; // 6.0
    std::cout << "bool                  = " << std::boolalpha << p2.second << "\n"; // false

    return 0;
}