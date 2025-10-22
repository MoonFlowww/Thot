#ifndef THOT_CORE_HPP
#define THOT_CORE_HPP
/*
 * This file correspond to the Core/Brain of the Wrapper, it will collect request from main.cpp, send orders to module-factories, and send function pointers to network.hpp.
 * In Order the keep network.hpp pure, with only what's needed to forward/backword the network
 * We must keep the runtime fast, by keeping "Do we use ..." in constexpr
 */

#endif //THOT_CORE_HPP