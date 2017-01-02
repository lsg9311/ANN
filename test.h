#pragma once
#include "data.h"
#include "network.h"

//test data classify
class TSC {
private:
	double s_rate;
	double e_rate;

	double T0;
	double T1;
	double F0; //false negative
	double F1; //false positive
public:
	TSC();
	void eval_rate(DataSet* test_set, double threshold, Neural_net* network);
	void init();

	double get_AC();
	double get_TP();
	double get_FP();

	void print_result();
};