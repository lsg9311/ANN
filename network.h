#pragma once
#include <vector>
#include "data.h"
#include "layer.h"

class Neural_net {
private:
	LPerceptron output; //linear output node
	vector<Layer> hidden; // W : weight matrix of hidden node
	int epock; //number of epock
	int* node_size; //number of nodes in each layer
	double alpha; //coef of CD update
public:
	//Creater
	Neural_net();

	//Allocator 
	void alloc_layer(int hlayer_size,int* node_size); //set number of layers and nodes in each layer

	//initialize
	void init_weight(); // initialize V,W with small random numbers
	void set_input(double* x);
	void set_rate(double eta,double alpha);
	void set_epock(int epock);
	void set_alpha(double alpha);

	//training
	void training(DataSet *data);
	void feed_forward(double* x);
	void EBP(double r);

	//get
	int get_output(double threshold);

	//CD
	void CD(DataSet *data);
};