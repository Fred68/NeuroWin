
#include "layer.h"
#include "network.h"

namespace neuro
{
	class network;

	void layer::reset()
	{
		for(uint i=0; i<_neurons.size(); i++)
		{
			_neurons[i].reset();
		}
		_neurons.clear();
	}

	#if _DEBUG_DTOR_LAY
	layer::~layer()
	{
		//reset();
		std::cout << "~layer()\n";
	}
	#endif

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
		//_neurons.resize(other._neurons.size(), { _net, false });
		for (uint i = 0; i < other._neurons.size(); i++)
		{
			_neurons.push_back(other._neurons[i]);
			//_neurons[i] = other._neurons[i];
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
			for (uint i = 0; i < _neurons.size(); i++)
			{
				_neurons[i].write(fs);
			}
		}
		catch(std::exception &ex)
		{
			std::cerr << "Eccezione exception in layer::write(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		}
		catch(neuro_exceptions::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in layer::write(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		}
	}

	void layer::read(std::ifstream &fs)
	{
		try
		{
			bool r_tmp;
			fs.read(reinterpret_cast<char*>(&r_tmp), sizeof(r_tmp));
			_recalc_w = r_tmp;

			for (uint i = 0; i < _neurons.size(); i++)
			{
				_neurons[i].read(fs);
			}
		}
		catch (std::exception &ex)
		{
			std::cerr << "Eccezione exception in layer::read(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		} catch (neuro_exceptions::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in layer::read(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		}


	}




	
	


}