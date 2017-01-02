#pragma once
#include <string>
#include <vector>
#include <fstream>

using namespace std;

struct Data {
	double x[13] = { 0 };
	int r;
	int index;
};

class DataSet {
private:
	vector<Data> data_list;
public:
	DataSet();
	void insert_data(Data new_data);
	Data get_data(int id);

	int size();

	void shuffle();
};

class DataReader {
public:
	void save_data(string filepath, DataSet* data_set);
};