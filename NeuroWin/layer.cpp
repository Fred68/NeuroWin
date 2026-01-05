
#include "layer.h"
#include "network.h"

namespace neuro
{
	class network;

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
			_neurons.resize(sz_tmp, { _net, true });

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