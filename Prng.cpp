/**
 * Prng.cpp
 */

#include "cuben.hpp"

cuben::Prng::Prng() {
    state = 1;
    nRolls = 0;
}

cuben::Prng::Prng(unsigned int s) {
    state = s;
}

float cuben::Prng::roll() {
    std::srand(state);
    nRolls += 1;
    return (float)std::rand() / RAND_MAX;
}

unsigned int cuben::Prng::getState() {
    return state;
}
