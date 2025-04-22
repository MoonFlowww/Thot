#include <iostream>

#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model;
	
	model.add(Thot::Layer::FC(2, 6, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model.summary();
	return 0;
}
