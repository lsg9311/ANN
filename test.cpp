#include "test.h"
#include <iostream>

using namespace std;

//Test data classifier
TSC::TSC() {
	this->s_rate = 0;
	this->e_rate = 0;
	this->T0 = 0;
	this->T1 = 0;
	this->F0 = 0;
	this->F1 = 0;
}
void TSC::init() {
	this->s_rate = 0;
	this->e_rate = 0;
	this->T0 = 0;
	this->T1 = 0;
	this->F0 = 0;
	this->F1 = 0;
}

double TSC::get_AC() {
	return (double)((this->T0 + this->T1) / (this->T0 + this->T1 + this->F0 + this->F1));
}

double TSC::get_TP() {
	return (double)((this->T1) / (this->T1 + this->F0));
}
double TSC::get_FP() {
	return (double)((this->F1) / (this->T0 + this->F1));
}

void TSC::eval_rate(DataSet* test_set,  double threshold, Neural_net* network) {
	int success = 0;
	int error = 0;
	int N = test_set->size();

	int answer;

	Data cur_data;

	for (int i = 0; i < N; i++) {
		cur_data = test_set->get_data(i);
		network->feed_forward(cur_data.x);
		answer=network->get_output(threshold);

		//test
		if (answer == cur_data.r) {
			success++;
			if (answer == 0) this->T0++;
			else if (answer == 1) this->T1++;

		}
		else {
			error++;
			if (answer == 0) this->F0++;
			else if (answer == 1) this->F1++;
		}
	}
	
	//evaluate
	this->s_rate = (double)success / (double)N;
	this->e_rate = (double)error / (double)N;
	return;
}

void TSC::print_result() {
	cout << "Success : " << this->s_rate << endl;
	cout << "Error : " << this->e_rate << endl;
	cout << "AC : " << (double)((this->T0 + this->T1) / (this->T0 + this->T1 + this->F0 + this->F1)) << endl;
	cout << "TP : " << (this->T1) << endl;
	cout << "TN : " << (this->T0) << endl;
	cout << "FP : " << (this->F1) << endl;
	cout << "FN : " << (this->F0) << endl;
	cout << "TPR : " << (double)((this->T1) / (this->T1 + this->F0)) << endl;
	cout << "TNR : " << (double)((this->T0) / (this->T0 + this->F1)) << endl;
	cout << "FPR : " << (double)((this->F1) / (this->T0 + this->F1)) << endl;
	cout << "FNR : " << (double)((this->F0) / (this->T1 + this->F0)) << endl;
	
	return;
}