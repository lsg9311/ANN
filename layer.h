#pragma once
#include "perceptron.h"


class Layer {
private:
	vector<SPerceptron> node;
	vector<double> node_result;
	int node_size;
public:
	//Creater
	Layer();

	//get
	vector<double> get_result();
	int size();
	Perceptron* get_node(int index);

	//set
	void set_node(int node_size);
	void alloc_w(int input_size);
	void set_rate(double eta,double alpha);

	//eval
	void eval_node(vector<double> input);
	void save_result();
};