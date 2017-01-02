#include "data.h"
#include "network.h"
#include "test.h"

#include <iostream>
#include <fstream>

#define HIDDEN_LAYER_NUM 6
#define INPUT_LAYER_NUM 13

using namespace std;

int main(){
	DataSet training_set = DataSet();
	DataSet test_set = DataSet();
	DataReader data_reader = DataReader();
	TSC test_model = TSC();

	int node_size[HIDDEN_LAYER_NUM] = { 10,10,10,10,10,INPUT_LAYER_NUM }; //include input_layer
	Neural_net neural_net = Neural_net();
	neural_net.alloc_layer(HIDDEN_LAYER_NUM, node_size);
	
	data_reader.save_data("trn.txt", &training_set);

	neural_net.init_weight();
	neural_net.set_rate(0.005,0.5);
	neural_net.set_epock(1000);
	neural_net.set_alpha(0.0001);
	cout << "Initialize Complete" << endl;

	neural_net.training(&training_set);
	cout << "Training Complete" << endl;

	data_reader.save_data("tst.txt", &test_set);
	test_model.eval_rate(&test_set, 0.5, &neural_net);
	test_model.print_result();
	
/*
	double threshold;
	struct Result {
		double TP;
		double FP;
	};
	vector<Result> eval_list = vector<Result>();
	for (threshold = 0; threshold <= 1; threshold += 0.01) {
		test_model = TSC();
		test_model.eval_rate(&test_set, threshold, &neural_net);
		Result cur_result;
		cur_result.TP = test_model.get_TP();
		cur_result.FP = test_model.get_FP();
		eval_list.push_back(cur_result);
	}

	ofstream outFile = ofstream("roc.txt");
	Result cur_result;
	outFile << "TP\tFP" << endl;
	for (int i = 0; i < eval_list.size(); i++) {
		cur_result = eval_list.at(i);
		outFile << cur_result.TP << "\t" << cur_result.FP << endl;
	}
	outFile.close();
	cout << "Test Complete" << endl;
	*/
}