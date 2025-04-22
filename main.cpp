#include <iostream>

#include <vector>
#include "Thot/Thot.hpp"

int main() {
	Thot::Network model1;
	model1.add(Thot::Transformer::Titan(Thot::Module::MoE, Thot::Normalization::DyT, Thot::Attention::MLA, Thot::Normalization::RMSE);
	
	model1.add(Thot::Layer::RBM(768, 256, Thot::Activation::LeakyReLU, Thot::Initialization::Xavier));
	model1.add(256, 256, Thot::Attention::MLA.layer(1))
	model1.add(Thot::Layer::RBM(256, 64, Thot::Activation::LeakyReLU, Thot::Initialization::Uniform));
	model1.add(256, 256, Thot::Attention::MLA.layer(1))
	model1.add(Thot::Layer::RBM(64, 16, Thot::Activation::LeakyReLU, Thot::Initialization::LeCun));
	
	model1.add(16, 16, Thot::Normalization::SoftMax, Thot::Penalization::ADF);
	
	model1.summary();


	Thot::Network model2;
	model2.add(Thot::Layer::FC(8, 8, Thot::Activation::ReLU, Thot::Initialization::Xavier));
	model2.add(Thot::Layer::Conv2D(8, 4, Thot::Activation::Sigmoid, Thot::Initialization::He));

	model2.add(Thot::Module::Finetune(4, 2, 5, Thot::Activation::ELU, Thot::Activation::SoftMax, THot::Initialization::Uniform)); // will be train in block-wise
	model2.set_optimizer(Thot::optimizations::SGDM(0.01f, 0.9f));

	
	model2.summary();

	Thot::Network GlobalModel;

	GlobalModel.ConnectBlock(16, 8, model1, model2, Thot::DimReduction::UMAP); // PCA, t-sne, UMAP
	GlobalModel.summary();
	GlobalModel.train(__data__);


	GlobalModel.predict(__data2__);
	GlobalModel.Eval(Thot::Metric::WQL, Thot::Metric::MSE, Thot::Metric::F1, Thot::Metric::ELO);
	GlobalModel.BlockFeaturesAnalysis(Thot::Analysis::DimReduction::PartialDependence);
	

	GlobalModel.save(__PATH__);
	
	
	return 0;
}
