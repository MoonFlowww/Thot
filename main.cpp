#include <iostream>

#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model;
	
	model.add(Thot::Layer::FC(2, 6, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model.add(Thot::Layer::FC(6, 6, Thot::Activation::Sigmoid, Thot::Initialization::Xavier));
	model.add(Thot::Layer::FC(6, 2, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model.set_optimizer(Thot::Optimizer::SGDM(0.001f, 0.85f)); 
	model.summary();
	return 0;
}
