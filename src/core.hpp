#ifndef THOT_CORE_HPP
#define THOT_CORE_HPP
/*
 * Core orchestrator of the framework.
 * ---------------------------------------------------------------------------
 * Planned responsibilities:
 *  - Define the master constexpr configuration (model topology, data pipeline,
 *    optimization, metrics, regularization) that remains immutable at runtime.
 *  - Collect requests from main.cpp and route them to the appropriate
 *    factories located under the module directories (activation, block, layer,
 *    optimizer, etc.).
 *  - Materialise the selected components as inline function objects or raw
 *    pointers that can be handed over to network.hpp without leaking
 *    higher-level orchestration details.
 *  - Push only the latency-inflating feature toggles (regularisation, data
 *    augmentation, k-folding, etc.) behind `if constexpr` / template
 *    specialisation so the generated runtime code path is branchless once the
 *    compile-time configuration is instantiated; constant-cost utilities can
 *    stay in regular functions for readability.
 *  - Expose helper APIs to retrieve training, evaluation, calibration and
 *    monitoring routines pre-bound to the compile-time configuration.
 */

// Upcoming API sketch (to be refined during implementation):
// namespace Thot::Core {
//     struct CompileTimeConfig;          // constexpr description of enabled features
//     constexpr auto make_runtime();     // builds the aggregate runtime facade
//     constexpr auto make_network();     // prepares callable forward/backward functions
//     constexpr auto make_trainer();     // orchestrates epoch/mini-batch execution
//     constexpr auto make_evaluator();   // returns evaluation functors
// }

#endif //THOT_CORE_HPP