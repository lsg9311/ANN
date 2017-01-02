#include "perceptron.h"
#include <iostream>
#include <math.h>

Perceptron::Perceptron() {
	this->result = 0;
	this->bias = 0;
	this->prev_bias = 0;
	this->weight = vector<double>();
	this->momentum = vector<double>();
	this->prev_info = vector<double>();
}
void Perceptron::alloc_w(int capacity) {
	for (int i=0; i < capacity; i++) {
		this->weight.push_back(0);
		this->momentum.push_back(0);
		this->prev_info.push_back(0);
	}
}

double Perceptron::get_result() {
	return this->result;
}
vector<double> Perceptron::get_momentum() {
	return this->momentum;
}
void Perceptron::set_result(double value) {
	this->result = value;
}

vector<double> Perceptron::get_weight() {
	return this->weight;
}
double Perceptron::get_weight(int index) {
	return this->weight.at(index);
}
void Perceptron::set_weight(vector<double> w) {
	this->weight = w;
}
void Perceptron::set_weight(double w, int index) {
	this->weight.at(index) = w;
}

double Perceptron::get_bias() {
	return this->bias;
}

void Perceptron::set_bias(double bias) {
	this->bias = bias;
}
void Perceptron::set_eta(double eta) {
	this->eta = eta;
}
void Perceptron::set_alpha(double alpha) {
	this->alpha = alpha;
}
double Perceptron::get_prev(int index) {
	return this->prev_info.at(index);
}

void Perceptron::update(double weight, int i) {
	double new_w;

	new_w = this->weight.at(i) + this->eta*weight + this->alpha*this->momentum.at(i);
	this->momentum.at(i) = this->eta*weight;

	this->set_weight(new_w, i);
}

void Perceptron::update(double pre_weight, double input, int index) {
	double weight,new_w;
	weight = pre_weight*input;
	new_w = this->weight.at(index) + this->eta*weight + this->alpha*this->momentum.at(index);
	this->momentum.at(index) = this->eta*weight;
	this->set_weight(new_w, index);
	
	this->prev_info.at(index) = pre_weight*this->weight.at(index);
}
void Perceptron::update_bias(double pre_weight) {
	this->bias += -1*this->eta*pre_weight+this->alpha*this->prev_bias;
	this->prev_bias = -1*this->eta*pre_weight;
}


void Perceptron::update_prev() {
	for(int i=0;i<this->prev_info.size();i++)
		this->prev_info.at(i) = this->result*this->weight.at(i);
}

void Perceptron::update_prev(bool active) {

	for (int i = 0; i<this->prev_info.size(); i++){
		this->prev_info.at(i) = active*this->weight.at(i);
	}
}

bool Perceptron::is_activated() {
	return (this->result >= this->bias) ? 1 : 0;
}

void LPerceptron::eval_result(vector<double> x) {
	if (this->weight.size() != x.size()) {
		cout << "Size is not matched" << endl;
	}
	double y=0;
	for (int i=0; i < this->weight.size(); i++) {
		y += x.at(i)*this->weight.at(i);
	}
	this->result = y;
}

int LPerceptron::get_output(double threshold) {
	return (this->result >= threshold) ? 1 : 0;
}

void SPerceptron::eval_result(vector<double> x) {
	if (this->weight.size() != x.size()) {
		cout << "Size is not matched" << endl;
	}
	double o = 0;
	for (int i=0; i < this->weight.size(); i++) {
		o+=x.at(i)*this->weight.at(i);
	}
	this->result = 1.0 / (1.0 + exp((-1 * o + this->bias)));
}
