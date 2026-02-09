


#include "neuron.h"



namespace neuro
{
    /*******************************************/
    /*                                         */
    /* synapse                                 */
    /*                                         */
    /*******************************************/

    neuron::synapse::synapse()
    {
		reset();
        #if _DEBUG_NEURO_DET
        std::cout << "synapse()\n";
        #endif
    }
	neuron::synapse::synapse(neuron &p_n, act ws)
    {
		_pn = std::shared_ptr<neuron>(&p_n);     // Non usa _pn=std::make_shared<neuron>(p_n) perché creerebbe una copia
		_in = UINT_ERROR;
        w = ws;
        #if _DEBUG_NEURO_DET
		std::cout << "synapse(p_n)\n";
        #endif
    }

	#if _DEBUG_DTOR
	neuron::synapse::~synapse()
    {
		reset();
        #if _DEBUG_NEURO_DET
		std::cout << "~synapse()\n";
        #endif
    }
	#endif

	void neuron::synapse::reset()
	{
		// Errato usare _pn = std::shared_ptr<neuron>(nullptr) oppure std::make_shared<neuron>()
		_pn.reset();
		_in = UINT_ERROR;
		w = (act)1;
	}

	#if _COPY_CTORS_
	neuron::synapse::synapse(const synapse& other) : _in{other._in}, _pn{other._pn}, w{other.w}
	{}
	neuron::synapse& neuron::synapse::operator=(const synapse& other)
	{
		_in = other._in;
		_pn = other._pn;
		w = other.w;
		return *this;
	}
	#endif
	#if _MOVE_CTORS_
	neuron::synapse::synapse(synapse&& other) : _in { other._in }, _pn{ other._pn }, w{ other.w }
	{}
	neuron::synapse& neuron::synapse::operator=(synapse&& other)
	{
		_in = other._in;
		_pn = other._pn;
		w = other.w;
		return *this;
	}
	#endif

	bool neuron::synapse::update_node_index()
	{
		bool ok = false;
		if(_pn.get() != nullptr)
		{
			_in = _pn->get_index();
			ok = true;
		}
		else
		{
			_in = UINT_ERROR;
		}
		return ok;
	}

	void neuron::synapse::set_node_index(uint i)
	{
		_in = i;
		_pn.reset();	// Non usare _pn = nullptr;
	}

	void neuron::synapse::write(std::ofstream &fs)
	{
		try
		{
			// Non scrive l'indice: già nella tolopogia
			// _in = _pn->get_index();
			// fs.write(reinterpret_cast<char*>(&_in), sizeof(_in));
			fs.write(reinterpret_cast<char*>(&w), sizeof(w));			// Scrive il peso
		}
		catch (std::exception &ex)
		{
			std::cerr << "Errore in synapse::write(...): " << ex.what() << std::endl;
		}
	}

	void neuron::synapse::read(std::ifstream &fs)
	{
		try
		{
			// Non scrive l'indice: rete già creata con la tolopogia
			//uint i_tmp;
			//fs.read(reinterpret_cast<char*>(&i_tmp), sizeof(i_tmp));
			//_in = i_tmp;
			//_pn.reset();

			act w_tmp;
			fs.read(reinterpret_cast<char*>(&w_tmp), sizeof(w_tmp));	// Legge il peso
			w = w_tmp;
		}
		catch (std::exception &ex)
		{
			std::cerr << "Errore in synapse::read(...): " << ex.what() << std::endl;
		}

	}

}