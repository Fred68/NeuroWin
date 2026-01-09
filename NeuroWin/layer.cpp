
#include "layer.h"
#include "network.h"

namespace neuro
{
	class network;

	#if _COPY_CTORS_
	layer::layer(const layer& other) : _neurons(other._neurons.size(), { other._net, true }), _net{ other._net }, _recalc_w{ other._recalc_w }
	{
		for(uint i=0; i<_neurons.size(); i++)
		{
			_neurons[i] = other._neurons[i];
		}
	}
	layer& layer::operator=(const layer& other)
	{
		_net = other._net;
		_recalc_w = other._recalc_w;
		_neurons.clear();
		_neurons.resize(other._neurons.size(), { _net, false });
		for (uint i = 0; i < other._neurons.size(); i++)
		{
			// _neurons.push_back(other._neurons[i]);
			_neurons[i] = other._neurons[i];
		}
		return *this;
	}
	#endif
	#if _MOVE_CTORS_
	layer::layer(const layer&& other) : _net{ other._net }, _recalc_w{ other._recalc_w }
	{
		_neurons = std::move(other._neurons);
	}
	layer& layer::operator=(const layer&& other)
	{
		_net = other._net;
		_recalc_w = other._recalc_w;
		_neurons = std::move(other._neurons);
		return *this;
	}
	#endif

	void layer::write(std::ofstream &fs)
	{
		try
		{
			fs.write(reinterpret_cast<char*>(&_recalc_w), sizeof(_recalc_w));
			size_t ssz = _neurons.size();
			fs.write(reinterpret_cast<char*>(&ssz), sizeof(ssz));
			for (uint i = 0; i < ssz; i++)
			{
				_neurons[i].write(fs);
			}
		}
		catch(std::exception &ex)
		{
			std::cerr << "Eccezione exception in layer::write(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		}
		catch(network::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in layer::write(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		}
	}

	void layer::read(std::ifstream &fs)
	{
		try
		{
			bool r_tmp;
			size_t sz_tmp;
			fs.read(reinterpret_cast<char*>(&r_tmp), sizeof(r_tmp));
			fs.read(reinterpret_cast<char*>(&sz_tmp), sizeof(sz_tmp));

			_recalc_w = r_tmp;
			_neurons.clear();
			_neurons.resize(sz_tmp, { _net, true });	// Ridimensiona, passando gli argomenti del costruttore di neuron

			for (uint i = 0; i < sz_tmp; i++)
			{
				fs.read(reinterpret_cast<char*>(&_neurons[i]), sizeof(neuron));
			}


		}
		catch (std::exception &ex)
		{
			std::cerr << "Eccezione exception in layer::read(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		} catch (network::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in layer::read(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		}


	}




	
	


}