#ifndef THOT_NETWORK_HPP
#define THOT_NETWORK_HPP
/*
 * This file correspond to the network-only function.
 * It must be if-free, or, if can't be avoided then if constexpr.
 * Network will send pointers of function (for every module that compose a Neural Network, e.g. layers/opt/loss/regularization/blocks/...)
 * Only what's needed for forward and backward of the total network will be coded in here
 * We must keep the runtime exempt of conditions such as "Do we use regularization", "Do we use kfold", "...". Or any non-necessary logic for the runtime of Forward&Backward
*/


#include <cstddef>
#include <tuple>
#include <utility>

#include <torch/torch.h>

#endif //THOT_NETWORK_HPP