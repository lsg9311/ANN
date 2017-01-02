#pragma once
#include <vector>

using namespace std;

class Perceptron {
protected:
	double result;
	vector<double> weight;
	double bias;
	double prev_bias;
	double eta;
	double alpha;
	vector<double> momentum;
	vector<double> prev_info;
public:
	Perceptron();
	void alloc_w(int capacity);

	double get_result();
	void set_result(double value);

	vector<double> get_weight();
	double get_weight(int index);
	void set_weight(vector<double> w);
	void set_weight(double w, int index);
	
	double get_bias();
	void set_bias(double bias);
	void set_eta(double eta);
	void set_alpha(double alpha);

	vector<double> get_momentum();
	double get_prev(int index);

	void update(double weight, int index);
	void update(double pre_weight, double input, int index);
	void update_bias(double pre_weight);

	//for CD
	void update_prev(); 
	void update_prev(bool active);
	bool is_activated();
};

class LPerceptron : public Perceptron{
public:
	void eval_result(vector<double> x);
	int get_output(double threshold);
};

class SPerceptron : public Perceptron {
public:
	void eval_result(vector<double> x);
};
