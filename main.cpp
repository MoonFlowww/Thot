#include <iostream>

#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model;

	model.add(Thot::Layer::FC(2, 4, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model.set_optimizer(Thot::optimizations::SGDM(0.01f, 0.9f));

	return 0;
}