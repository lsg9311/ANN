#include "layer.h"
//Creater
Layer::Layer() {
	this->node = vector<SPerceptron>();
	this->node_result = vector<double>();
	this->node_size = 0;
}

//get
vector<double> Layer::get_result() {
	return this->node_result;
}
int Layer::size() {
	return this->node_size;
}
Perceptron* Layer::get_node(int index) {
	return &(this->node.at(index));
}


//set
void Layer::set_node(int node_size) {
	this->node_size = node_size;
	SPerceptron new_node;
	for (int i = 0; i < node_size; i++) {
		new_node = SPerceptron();
		this->node.push_back(new_node);
		this->node_result.push_back(0);
	}
	return;
}
void Layer::set_rate(double eta, double alpha) {
	Perceptron* cur_node;
	for (int i = 0; i < this->node.size(); i++) {
		cur_node = &this->node.at(i);
		cur_node->set_eta(eta);
		cur_node->set_alpha(alpha);
	}
}
void Layer::alloc_w(int input_size) {
	SPerceptron* cur_node;
	for (int i = 0; i < this->node.size(); i++) {
		cur_node = &this->node.at(i);
		cur_node->alloc_w(input_size);
	}
}
//eval
void Layer::eval_node(vector<double> input) {
	SPerceptron* cur_node;
	for (int i = 0; i < this->node.size(); i++) {
		cur_node = &this->node.at(i);
		cur_node->eval_result(input);
	}
	this->save_result();
}

void Layer::save_result() {
	SPerceptron* cur_node;
	for (int i = 0; i < this->node.size(); i++) {
		cur_node = &this->node.at(i);
		this->node_result.at(i)=cur_node->get_result();
	}
}

