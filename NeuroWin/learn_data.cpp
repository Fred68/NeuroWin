
#include "neuro_def.h"
#include "learn_data.h"
#include <tuple> 


namespace neuro
{

	learn_data::learn_data(std::shared_ptr<network> pnet) :
		_pnet(pnet),
		_inp_sz(pnet->get_input_layer_size()),
		_out_sz(pnet->get_output_layer_size()
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
		if ((index_input != neuro::UINT_ERROR) && (index_output != neuro::UINT_ERROR))
		{
			_ldata.emplace_back(index_input,index_output);
		}
		else
		{
			throw _pnet->create_exception(network::neuro_exception::learn_data_index,true,"in learn_data::add_data()");
		}
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
		std::vector<act> &vout = get_output(std::get<1>(idat));
		return std::tuple<std::vector<act>&, std::vector<act>&>(vinp,vout);
	}

	std::vector<act> &learn_data::Iterator::get_input_v()
	{
		std::tuple<uint, uint> idat;
		try
		{
			idat = _ld._ldata.at(_indx);
		}
		catch (std::exception const &ex)
		{
			std::cerr << ex.what() << std::endl;
		}
		return _ld.get_input(std::get<0>(idat));
	}
	std::vector<act> &learn_data::Iterator::get_output_v()
	{
		std::tuple<uint, uint> idat;
		try
		{
			idat = _ld._ldata.at(_indx);
		}
		catch (std::exception const &ex)
		{
			std::cerr << ex.what() << std::endl;
		}
		return _ld.get_output(std::get<1>(idat));
	}

	bool learn_data::check_data_size()
	{
		bool ok = true;
		for (auto it = begin(); it != end(); it++)
		{
			uint visz = it.get_input_v().size();
			uint vosz = it.get_output_v().size();
			if ( (visz != _pnet->get_input_layer_size())||(vosz != _pnet->get_output_layer_size()) )
			{
				ok = false;
				break;
			}
		}
		return ok;
	}

}
