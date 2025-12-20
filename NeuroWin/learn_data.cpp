
#include "neuro_def.h"
#include "learn_data.h"
//#include "network.h"
#include <tuple> 


namespace neuro
{

	learn_data::learn_data(std::shared_ptr<network> pnet) :
		_pnet(pnet),
		_inp_sz(pnet->get_input_layer_sz()),
		_out_sz(pnet->get_output_layer_sz()
		)
	{}

	std::string learn_data::to_string()
	{
		std::string txt;
		txt += std::format("Vector lengths (input/output): {0}/{1}\n", _inp_sz, _out_sz);
		txt += std::format("Vector numbers (input/output): {0}/{1}\n", _vinp.size(), _vout.size());
		txt += std::format("Data pairs number: {0}", _vinp.size(), _ldata.size());

		return txt;
	}


	uint learn_data::add_input(std::vector<act> v)
	{
		uint indx = UINT_ERROR;
		if(v.size()==_inp_sz)
		{
			_vinp.push_back(v);
			indx = _vinp.size()-1;
		}
		return indx;
	}
	uint learn_data::add_output(std::vector<act> v)
	{
		uint indx = UINT_ERROR;
		if (v.size() == _out_sz)
		{
			_vout.push_back(v);
			indx = _vout.size() - 1;
		}
		return indx;

	}
	void learn_data::add_data(uint index_input, uint index_output)
	{
		_ldata.emplace_back(index_input,index_output);
	}

	std::vector<act> &learn_data::get_input(uint i)
	{
		return _vinp.at(i);
	}
	std::vector<act> &learn_data::get_output(uint i)
	{
		return _vout.at(i);
	}
	
	std::tuple<std::vector<act>&, std::vector<act>&> learn_data::get_data(uint i)
	{
		std::tuple<uint, uint> idat = _ldata.at(i);
		std::vector<act> &vinp = get_input(std::get<0>(idat));
		std::vector<act> &vout = get_input(std::get<1>(idat));
		return std::tuple<std::vector<act>&, std::vector<act>&>(vinp,vout);
	}


}
