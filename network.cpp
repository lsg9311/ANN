#include "network.h"

#include <stdlib.h>     
#include <time.h>
#include <math.h>
#include <iostream>

Neural_net::Neural_net() {
	this->output = LPerceptron();
	this->hidden = vector<Layer>();
}

void Neural_net::alloc_layer (int hLayer_size,int* node_size) { //alloc layer
	int iter;
	Layer new_layer;
	//output layer
	this->output = LPerceptron();
	this->output.alloc_w(node_size[0]);

	//hidden layer
	this->node_size = node_size;
	for (iter = 0; iter < hLayer_size; iter++) {
		new_layer = Layer();
		new_layer.set_node(node_size[iter]);
		if(iter<hLayer_size-1){
			new_layer.alloc_w(node_size[iter + 1]);
		}
		this->hidden.push_back(new_layer);
	}
}

void Neural_net::init_weight() { // initialize V,W with small random numbers
	int iter,iter1,iter2;
	double init_num = 0;
	double init_bias = 0;
	Perceptron* cur_node;
	Layer* cur_layer;
	vector<double> cur_weight = vector<double>();
	vector<double> V = vector<double>();

	srand((unsigned)time(NULL));
	//update V
	for (iter = 0; iter <node_size[0]; iter++) {
		init_num = (((double)rand() / RAND_MAX)/500.0) - 0.001; // -0.001 ~ +0.001
		V.push_back(init_num);
	}
	output.set_weight(V);

	//update W except output layer
	for (iter = 0; iter < this->hidden.size() - 1; iter++) {
		cur_layer = &(this->hidden.at(iter));
		for (iter1 = 0; iter1 < cur_layer->size(); iter1++) {
			cur_node = cur_layer->get_node(iter1);
			//init bias
			init_bias = (((double)rand() / RAND_MAX) / 500.0) - 0.001;
			cur_node->set_bias(init_bias);
			//init weight
			cur_weight = cur_node->get_weight();
			for (iter2 = 0; iter2 < cur_weight.size(); iter2++) {
				init_num = (((double)rand() / RAND_MAX) / 500.0) - 0.001; // -0.001 ~ +0.001
				cur_weight.at(iter2) = init_num;
			}
			cur_node->set_weight(cur_weight);
		}
	}
}

void Neural_net::set_rate(double eta,double alpha) {
	this->output.set_eta(eta);
	this->output.set_alpha(alpha);
	for (int i = 0; i < this->hidden.size()-1; i++) {
		this->hidden.at(i).set_rate(eta,alpha);
	}
}

void Neural_net::set_alpha(double alpha) {
	this->alpha = alpha;
}
void Neural_net::set_epock(int epock) {
	this->epock = epock;
}


void Neural_net::set_input(double* x){
	Layer* input_layer = &this->hidden.at(this->hidden.size() - 1);
	if (input_layer->size() != 13) {
		cout << "Input size is not equal" << endl;
		return;
	}
	Perceptron* cur_node;
	for (int i = 0; i < 13; i++) {
		cur_node = input_layer->get_node(i);
		cur_node->set_result(x[i]);
	}
	input_layer->save_result();
}

void Neural_net::training(DataSet *data) {
	Data cur_data;
	int N, cur_index;
	double error = 0;
	
	///CD///
	for(int i=0;i<10;i++){
		data->shuffle();
		this->CD(data);
	}
	cout << "CD complete"<< endl;
	
	for (int i = 0; i < epock; i++) {
		error = 0;
		data->shuffle();
		if (i == 50) this->set_rate(0.001, 0.9);
		//else if (i == 150) this->set_rate(0.001, 0.95);

		for(int j=0;j<data->size();j++){
			cur_data = data->get_data(j);
			this->feed_forward(cur_data.x);
			error += pow(cur_data.r - this->output.get_result(),2)/2;
			this->EBP(cur_data.r);
		}
		cout << i + 1 << "번째 epock 완료, Error : "<<error/data->size() << endl;
	}
}

void Neural_net::feed_forward(double* x) {
	int iter;
	Layer* cur_layer;
	Layer bot_layer;
	vector<double> input = vector<double>();
	//set input
	this->set_input(x);
	//eval hidden
	for (iter = this->hidden.size() - 2; iter >= 0; iter--) {
		cur_layer = &this->hidden.at(iter);
		bot_layer = this->hidden.at(iter + 1);
		input = bot_layer.get_result();
		cur_layer->eval_node(input);
	}
	//eval output
	this->output.eval_result(this->hidden.at(0).get_result());
}

void Neural_net::EBP(double r) {
	double y,cur_weight,pre_weight;
	y = this->output.get_result();
	Layer *top_layer,*cur_layer;
	Perceptron *cur_node, *pre_node;
	vector<double> x;
	///update V///
	int iter,iter1,iter2;
	x = this->hidden.at(0).get_result();
	for (iter = 0; iter < this->hidden.at(0).size(); iter++) {
		cur_weight = (r - y)*x.at(iter);
		this->output.update(r-y,x.at(iter), iter);
	}
	cur_layer = &this->hidden.at(0);
	///update W///
	for (iter = 0; iter < this->hidden.size() - 1; iter++) {
		top_layer = cur_layer;
		cur_layer = &this->hidden.at(iter);
		x = this->hidden.at(iter + 1).get_result();
		/// iter1 : current update node ///
		for (iter1 = 0; iter1 < cur_layer->size(); iter1++) {
			cur_node = cur_layer->get_node(iter1);
			pre_weight = 0;
			if (iter == 0) {
				pre_weight = (r - y)*this->output.get_weight().at(iter1);
			}
			else {
				/// iter2 : previous node connected with cur_node ///
				for (iter2 = 0; iter2 < top_layer->size(); iter2++) {
					pre_node = top_layer->get_node(iter2);
					pre_weight += pre_node->get_prev(iter1);
				}
			}
			pre_weight *= cur_node->get_result()*(1 - cur_node->get_result());
			/// iter2 : input node connected with cur_node ///
			for (iter2 = 0; iter2 < x.size(); iter2++) {
				cur_node->update(pre_weight, x.at(iter2),iter2);
			}
			cur_node->update_bias(pre_weight);
		}
	}
}

int Neural_net::get_output(double threshold) {
	return this->output.get_output(threshold);
}


/// Contrastive Divergence ///
void Neural_net::CD(DataSet* data) {
	Data cur_data;
	int cur_index,iter_h,iter_v,iter_d;
	vector<double> V,H,V_model,H_model;
	Layer *cur_layer,*bot_layer;
	SPerceptron *cur_node, *bot_node;
	double cur_weight,cur_bias,bot_bias,v_data,h_data,v_model,h_model,cur_prob,rand_num;
	for (iter_d = 0; iter_d<data->size(); iter_d++) {
		cur_data = data->get_data(iter_d);
		this->set_input(cur_data.x);
		for (cur_index = this->hidden.size() - 2; cur_index >= 0; cur_index--) {
			cur_layer = &this->hidden.at(cur_index);
			bot_layer = &this->hidden.at(cur_index+1);
			V = vector<double>();
			H = vector<double>();
			
			///eval v_data///
			for (iter_v = 0; iter_v < bot_layer->size(); iter_v++) {
				bot_node = (SPerceptron*)bot_layer->get_node(iter_v);
				v_data = (double)bot_node->is_activated();
				V.push_back(v_data);
			}

			///eval h_data//
			for (iter_h = 0; iter_h < cur_layer->size(); iter_h++) {
				cur_node = (SPerceptron*)cur_layer->get_node(iter_h);
				cur_node->eval_result(V);
				cur_node->update_prev();
			}
			cur_layer->save_result();
			H = cur_layer->get_result();

			V_model = vector<double>();
			H_model = vector<double>();
			///make random h activation///
			srand((unsigned)time(NULL));
			for (iter_h = 0; iter_h < cur_layer->size(); iter_h++) {
				cur_node = (SPerceptron*)cur_layer->get_node(iter_h);
				cur_prob = cur_node->get_result();
				rand_num = rand()%2;
				if (rand_num < cur_prob) cur_node->update_prev(0);
				else cur_node->update_prev(1);				
			}
			///eval v_model///
			for(iter_v=0;iter_v<bot_layer->size();iter_v++){
				v_model = 0;
				for (iter_h = 0; iter_h < cur_layer->size(); iter_h++) {
					cur_node = (SPerceptron*)cur_layer->get_node(iter_h);
					v_model += cur_node->get_prev(iter_v);
				}
				V_model.push_back(v_model);
			}
			///eval h_model///
			for (iter_h = 0; iter_h < cur_layer->size(); iter_h++) {
				cur_node = (SPerceptron*)cur_layer->get_node(iter_h);
				cur_node->eval_result(V_model);
				cur_node->update_prev();
			}
			cur_layer->save_result();
			H_model = cur_layer->get_result();	
			///weight update///
			for (iter_h = 0; iter_h < cur_layer->size(); iter_h++) {
				cur_node = (SPerceptron*)cur_layer->get_node(iter_h);
				for (iter_v = 0; iter_v < bot_layer->size(); iter_v++) {
					cur_weight = cur_node->get_weight(iter_v);
					v_data = V.at(iter_v);
					h_data = H.at(iter_h);
					v_model = V_model.at(iter_v);
					h_model = H_model.at(iter_h);

					cur_weight += this->alpha*(v_data*h_data - v_model*h_model);
					cur_node->set_weight(cur_weight, iter_v);
				}
			}
			/// bias update ///
			for (iter_h = 0; iter_h < cur_layer->size(); iter_h++) {
				cur_node = (SPerceptron*)cur_layer->get_node(iter_h);
				cur_bias = cur_node->get_bias();
				h_data = H.at(iter_h);
				h_model = H_model.at(iter_h);
				cur_bias += this->alpha*(h_data - h_model);
				cur_node->set_bias(cur_bias);
			}
			for (iter_v = 0; iter_v < bot_layer->size(); iter_v++) {
				bot_node = (SPerceptron*)bot_layer->get_node(iter_v);
				bot_bias = bot_node->get_bias();
				v_data = V.at(iter_v);
				v_model = V_model.at(iter_v);
				bot_bias += this->alpha*(v_data - v_model);
				bot_node->set_bias(bot_bias);
			}
		}
	}
}