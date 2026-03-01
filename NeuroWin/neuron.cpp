
#include "neuron.h"
#include "network.h"		// Nesessario, se no tipo incompleto "neuro::network" non consentito
#include <variant>

namespace neuro
{
    /*******************************************/
    /*                                         */
    /* neuron                                  */
    /*                                         */
    /*******************************************/

	// TODO : Prevedere cancellazione e creazione di sinapsi (alcuni livelli potrebbero avere solo poche sinapsi al livello precedente).
	// TODO : Vedere se e quando disabilitare dei nodi (se funzione relu < 0), ma probabilmente è meglio di no.

    neuron::neuron(network &netwrk) : _net{netwrk}
    {
		reset();

        #if _DEBUG_NEURO_DET
		std::cout << "neuron()\n";
        #endif
    }
	neuron::neuron(network &netwrk, bool isInput) : neuron{netwrk}
	{
		if(isInput)					// Se è un neurone di _input, imposta la funzione di attivazion identità  (ed il flag)
		{
			set_fact(FACT::id);		// Prima imposta FACT...
			_input = true;			// ...poi imposta _input a true, che disabilita set_fact()
		}
	}

	neuron::neuron(network &netwrk, std::vector<neuron> &prev, act neu_w, act bias_w) : neuron(netwrk)
    {
		for(uint i=0; i<prev.size(); i++)						// Imposta il vettore delle sinapsi (non è un neurone di _input)
		{														// con pesi e bias indicati
			neuron &n = prev[i];
			_syns.push_back(synapse(n, (i == prev.size() - 1) ? bias_w : neu_w));
		}
        #if _DEBUG_NEURO_DET
		std::cout << "neuron(neuron &prev)\n";
        #endif
    }

	neuron::neuron(network &netwrk, std::vector<neuron> &prev, std::vector<uint> &indx, act neu_w, act bias_w) : neuron(netwrk)
    {
		for (uint j = 0; j < indx.size(); j++)					// Percorre il vettore degli indici dei neuroni
		{														
			uint i = indx[j];
			if(i < prev.size())
			{
				neuron &n = prev[i];
				_syns.push_back(synapse(n, (i == prev.size() - 1) ? bias_w : neu_w));
			}
		}
        #if _DEBUG_NEURO_DET
		std::cout << "neuron(neuron &prev)\n";
        #endif
    }

	#if _COPY_CTORS_
	neuron::neuron(const neuron& other) :	_net{other._net},
											x{other.x}, y{other.y},
											f_act{other.f_act}, f_act_der{other.f_act_der}, _fact{other._fact},
											_active{other._active}, _input{other._input},
											_nstat{other._nstat}
											//_syns(other._syns.size())
	{
		switch(other._nstat)
		{
			case stat::_beta:
				beta = other.beta;
			break;
			case stat::_ei:
				ei = other.ei;
			break;
			case stat::_index:
				index_in_layer = other.index_in_layer;
			break;
			default:
				// TODO aggiungere throw eccezione
			break;
		}
		for(uint i=0; i<other._syns.size(); i++)
		{
			//_syns[i] = other._syns[i];
			// TODO I riferimenti delle sinapsi vanno aggiornate ai nodi della nuova rete
			_syns.push_back(synapse(other._syns[i]));
		}
	}
	neuron& neuron::operator=(const neuron& other)
	{
		_net = other._net;
		x = other.x;
		y = other.y;
		f_act = other.f_act;
		f_act_der = other.f_act_der;
		_fact = other._fact;
		_active = other._active;
		_input = other._input;
		_nstat = other._nstat;
		switch (other._nstat)
		{
		case stat::_beta:
			beta = other.beta;
			break;
		case stat::_ei:
			ei = other.ei;
			break;
		case stat::_index:
			index_in_layer = other.index_in_layer;
			break;
		default:
			// TODO aggiungere throw eccezione
			break;
		}
		_syns.clear();
		//_syns.resize(other._syns.size());
		for (uint i = 0; i < other._syns.size(); i++)
		{
			//_syns[i] = other._syns[i];
			// TODO I riferimenti delle sinapsi vanno aggiornate ai nodi della nuova rete
			_syns.push_back(synapse(other._syns[i]));
		}
		return *this;
	}
	#endif
	#if _MOVE_CTORS_
	neuron::neuron(neuron&& other) :	_net{ other._net },
										x{ other.x }, y{ other.y },
										f_act{ other.f_act }, f_act_der{ other.f_act_der }, _fact{ other._fact },
										_active{ other._active }, _input{ other._input },
										_nstat{ other._nstat }
	{
		switch (other._nstat)
		{
		case stat::_beta:
			beta = other.beta;
			break;
		case stat::_ei:
			ei = other.ei;
			break;
		case stat::_index:
			index_in_layer = other.index_in_layer;
			break;
		default:
			// TODO aggiungere throw eccezione
			break;
		}
		_syns.clear();
		_syns = std::move(other._syns);
	}
	neuron& neuron::operator=(neuron&& other)
	{
		_net = other._net;
		x = other.x;
		y = other.y;
		f_act = other.f_act;
		f_act_der = other.f_act_der;
		_fact = other._fact;
		_active = other._active;
		_input = other._input;
		_nstat = other._nstat;
		switch (other._nstat)
		{
		case stat::_beta:
			beta = other.beta;
			break;
		case stat::_ei:
			ei = other.ei;
			break;
		case stat::_index:
			index_in_layer = other.index_in_layer;
			break;
		default:
			// TODO aggiungere throw eccezione
			break;
		}
		_syns.clear();
		_syns = std::move(other._syns);
		return *this;
	}
	#endif

	#if _DEBUG_DTOR
    neuron::~neuron()
    {
		// vector<synapse> non ha bisogno di dtor.
		// TODO Prova azzeramento puntatori...
		/*for(uint i=0; i<_syns.size(); i++)
		{
			_syns[i].set_node_ptr(nullptr);
		}*/
		// No: usare sgared_ptr<T>.reset();
		reset();
        #if _DEBUG_NEURO_DET
		std::cout << "~neuron()\n";
        getchar();
        #endif
    }
	#endif   
	void neuron::reset()
	{
		x = y = 0;
		ei = 0;
		set_fact(fact_default());
		_active = true;
		_input = false;
		for(uint i=0; i<_syns.size(); i++)
		{
			_syns[i].reset();
		}
		_syns.clear();
	}

    std::string neuron::to_string()
    {
        std::string statStr = "";
		std::string type_stat = "?";
        if(!_active) statStr = "X";
		if(_input)  statStr += "I";
	
		switch(_nstat)
		{
			case stat::_beta:
				type_stat = "beta";
				break;
			case stat::_ei:
				type_stat = "ei";
				break;
			case stat::_index:
				type_stat = "index";
				break;
		}

		if(!statStr.empty())	statStr = "["+statStr+"]";

		std::string txt;

		if(_nstat == stat::_index)
		{
			txt = std::format(to_string_frm_n_indx, x, y, get_fact_name(), statStr, index_in_layer, type_stat);
		}
		else
		{
			txt = std::format(to_string_frm_n, x, y, get_fact_name(), statStr, ei, type_stat);
		}

        if(_active)
        {
            for(synapse &s : _syns)				// Ciclo su reference, se no chiama il copy ctor.
            {
                //if(s._pn.get() != nullptr)			// if (std::get<ptN>(s._pn) != nullptr)
				if (s._pn != nullptr)			// if (std::get<ptN>(s._pn) != nullptr)
                {
                    std::string nn = "";
                    #if TXT_INFO
                    nn = s._pn->get_name()+",";
                    #endif
                    txt = txt + std::format(to_string_frm_w, nn, s.w);
                }
            }
        }
        #if TXT_INFO
        txt = name + ": " + txt;
        #endif
        return txt;
    }    
 
	void neuron::set_active(bool stat)	{ _active = stat; }
	void neuron::set_input(bool inp)	{ _input = inp; }

	void neuron::set_fact(FACT f)
	{
		if(!_input)			// Se è un neurone di _input, la funzione di attivazione è quella definita nel costrutture
		{
			switch (f)
			{
			case FACT::sigmoid:
				f_act = &sigmoid;
				f_act_der = &sigmoid_der;
				break;
			case FACT::tanh:
				f_act = &hyptangent;
				f_act_der = &hyptangent_der;
				break;
			case FACT::relu:
				f_act = &relu;
				f_act_der = &relu_der;
				break;
			case FACT::one:
			case FACT::bias:
				f_act = &one;
				f_act_der = &zero;
				break;
			case FACT::id:
				f_act = &id;
				f_act_der = &one;
				break;
			default:
				throw _net.get_exceptions().create_exception(neuro_exceptions::type::activation_function, true, "in neuron::set_fact()");
				
			}
			_fact = f;
		}
	}
	std::string neuron::get_fact_name()
	{
		return fact2string(_fact);
	}
	
    bool neuron::set_x(act x_in)
    {
        if(_input)
        {
            x = x_in;
            return true;
        }
        return false;            
    }
	void neuron::set_ei(act ei_in)
	{
		ei = ei_in;
		_nstat = stat::_ei;
	}
	void neuron::set_beta(act beta_in)
	{
		beta = beta_in;
		_nstat = stat::_beta;
	}
	void neuron::set_index(uint indx)
	{
		index_in_layer = indx;					// Indice del neurone nel livello
		for(uint i=0; i<_syns.size(); i++)
		{
			if(!_syns[i].update_node_index())			// Indici dei nodi delle sinapsi
				throw _net.get_exceptions().create_exception(neuro_exceptions::type::null_pointer_synapse, true, "failed neuron::set_index(), synapse pointer to node not set");
		}
		_nstat = stat::_index;
	}

	act neuron::get_ei()
	{	
		if(_nstat!=stat::_ei)
			throw _net.get_exceptions().create_exception(neuro_exceptions::type::EI_mismatch, true, "failed neuron::get_ei(), EI is not");
		return ei;
		
	}
	act neuron::get_beta()
	{
		if (_nstat != stat::_beta)
			throw _net.get_exceptions().create_exception(neuro_exceptions::type::beta_mismatch, true, "failed neuron::get_beta(), beta is not set");
		return beta;
	}
	uint neuron::get_index()
	{
		if (_nstat != stat::_index)
			throw _net.get_exceptions().create_exception(neuro_exceptions::type::beta_mismatch, true, "failed neuron::get_index(), index is not set");
		return index_in_layer;
	}

	act neuron::get_w(uint i)
	{
		return (i < _syns.size()) ? _syns[i].w : 0;
	}

	uint neuron::get_neuron_index(uint i)
	{
		return (i < _syns.size()) ? _syns[i]._in : UINT_ERROR;
	}

	void neuron::set_w(act w, uint i)
	{
		if(i < _syns.size())
			_syns[i].w = w;
	}

	void neuron::add_synapse(uint in)
	{
		//synapse *s = new synapse();
		//s->set_node_index(in);
		//_syns.push_back(*s);
		_syns.push_back(synapse());
		_syns.back().set_node_index(in);
	}

    void neuron::calc_x()
    {
        if(_active && !_input)
        {
            #ifdef ACT_DBL
				std::atomic<act> sum = 0.0;
            #else
				std::atomic<act> sum = 0.0f;
            #endif
        
			//this;
			#if _SEQ_CYCLE
			for(uint i=0; i<_syns.size(); i++)
			{
				act yy = _syns[i]._pn->get_y();
				act ww = _syns[i].w;
				sum += yy * ww;
			}
			#else
            // Calcola, su tutte le sinapsi del neurone, la somma delle uscite y dei nodi collegati, moltiplicate...
            // ...per il peso w della sinapsi. Il risultato è il segnale di ingresso x del neurone.        
            // auto func_add = [&](const synapse &s) {sum.fetch_add(s._pn->y * s.w);};
			auto func_add = [&](const synapse &s) {sum.fetch_add(s._pn->get_y() * s.w); };
			//auto func_add = [&](const synapse &s) {sum.fetch_add(std::get<ptN>(s._pn)->y * s.w); };
            //std::for_each(_net.get_exe_pol(EXE_POL::neuron), _syns.begin(), _syns.end(), func_add);
			std::for_each(EXEPOL, _syns.begin(), _syns.end(), func_add);
			#endif

			this->x = sum;

			// TODO!!! Togliere... solo per debug
			//std::string sss = this->_net.to_string();
			//std::cout << sss;

			//std::cout << "x= " << sum << std::endl;
        }
    }

    void neuron::calc_y()
    {
        if(_active)
			y = f_act(this);
    }
	void neuron::calc_ei()
	{
		if(_active)
		{
			if (_nstat != stat::_beta)
				throw _net.get_exceptions().create_exception(neuro_exceptions::type::beta_mismatch, true, "failed neuron::calc_ei(), beta is not set");
			set_ei(get_beta() * f_act_der(this));
		}
		else
		{
			set_ei((act)0.0);
		}
	}
	void neuron::calc_parz_eai()
	{
		if (_active && !_input)
		{
			auto func_ea =   [&](const synapse &s) {s._pn->set_beta(s.w * get_ei());};
			//auto func_ea = [&](const synapse &s) {std::get<ptN>(s._pn)->set_beta(s.w * get_ei()); };
			std::for_each(EXEPOL, _syns.begin(), _syns.end(), func_ea);
		}
	}
	void neuron::calc_w(act learn_const)
	{
		if (_active && !_input)
		{
			auto func_updw = [&](synapse &s)
			{	
				// Corregge il peso wi della sinapsi tra in neurone j attuale e il neurone i precedente
				// con le formule [8] e [10], usando il prodotto tra ei (del neurone j) e y (del neurone i).
				//s.w -=  learn_const * ei * std::get<ptN>(s._pn)->y;
				s.w -= learn_const * ei * s._pn->y;
			};
			std::for_each(EXEPOL, _syns.begin(), _syns.end(), func_updw);
		}
	}

	/*******************************************/
    // Funzioni di attivazione
	/*******************************************/

    act neuron::sigmoid(neuron *n)
    {
        #ifdef ACT_DBL
            return 1.0 / (1.0 + std::exp(-n->x));
        #else
            return 1.0f / (1.0f + std::expf(-n->x));
        #endif
    }
    act neuron::sigmoid_der(neuron *n)
    {
        #ifdef ACT_DBL
            return n->y * (1.0 - n->y);
        #else
            return n->y * (1.0f - n->y);
        #endif    
    }
    act neuron::hyptangent(neuron *n)
    {
        #ifdef ACT_DBL
            return std::tanh(n->x);
        #else
            return std::tanhf(n->x);
        #endif
    }
    act neuron::hyptangent_der(neuron *n)
    {
        #ifdef ACT_DBL
            return 1.0 - n->y * n->y;
        #else
            return 1.0f - n->y * n->y;
        #endif    
    }
    act neuron::relu(neuron *n)
    {
        #ifdef ACT_DBL
            return (n->x > 0) ? n->x : 0.0;
        #else
            return (n->x > 0) ? n->x : 0.0f;
        #endif
    }
    act neuron::relu_der(neuron *n)
    {
        #ifdef ACT_DBL
            return (n->x > 0) ? 1.0 : 0.0;
        #else
            return (n->x > 0) ? 1.0f : 0.0f;
        #endif    
    }
    act neuron::one(neuron *n)
    {
        #ifdef ACT_DBL
            return 1.0;
        #else
            return 1.0f;
        #endif
    }
    act neuron::zero(neuron *n)
    {
        #ifdef ACT_DBL
            return 0.0;
        #else
            return 0.0f;
        #endif
    }
    act neuron::id(neuron *n)
    {
        #ifdef ACT_DBL
            return n->x;
        #else
            return n->x;
        #endif
    }

	void neuron::write(std::ofstream &fs)
	{
		try
		{
			if (_nstat == stat::_index)
			{
				fs.write(reinterpret_cast<char*>(&_fact), sizeof(_fact));
				fs.write(reinterpret_cast<char*>(&_active), sizeof(_active));
				fs.write(reinterpret_cast<char*>(&_input), sizeof(_input));
				for(uint i=0; i< _syns.size(); i++)
				{
					_syns[i].write(fs);
				}
				#if TXT_INFO
				size_t len = name.length();
				fs.write(reinterpret_cast<char*>(&len), sizeof(size_t));
				fs.write(name.c_str(), sizeof(char) * len);
				//fs.write(reinterpret_cast<char*>(&name), sizeof(name));
				#endif
			}
			else
			{
				throw _net.get_exceptions().create_exception(neuro_exceptions::type::index_mismatch, true, "Neuron index is not set");
			}
		}
		catch(std::exception &ex)
		{
			std::cerr << "Eccezione exception in neuron::write(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		}
		catch (neuro_exceptions::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in neuron::write(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		}
	}

	void neuron::read(std::ifstream &fs)
	{
		try
		{
			FACT f_tmp;
			bool active_tmp, input_tmp;

			fs.read(reinterpret_cast<char*>(&f_tmp), sizeof(f_tmp));
			fs.read(reinterpret_cast<char*>(&active_tmp), sizeof(active_tmp));
			fs.read(reinterpret_cast<char*>(&input_tmp), sizeof(input_tmp));

			set_fact(f_tmp);

			_active = active_tmp;
			_input = input_tmp;
			set_beta();				// Azzera
			
			for (uint i = 0; i < _syns.size(); i++)
			{
				_syns[i].read(fs);
			}
			#if TXT_INFO
			size_t len;
			fs.read(reinterpret_cast<char*>(&len), sizeof(size_t));
			name.resize(len);
			fs.read(&name[0], sizeof(char)*len);
			#endif

			if (get_fact() == FACT::bias)		// Se nodo di bias:...
			{
				set_active(true);
				calc_y();                      // Calcola l'uscita
				set_active(false);
			}

			
		}
		catch (std::exception &ex)
		{
			std::cerr << "Eccezione exception in neuron::read(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		} catch (neuro_exceptions::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in neuron::read(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) _net.create_exception...
		}
	}
	void neuron::update_syn_pointers(uint ilay/*, network &net_ref*/)
	{
		// Dopo set() i neuroni del livello 0 hanno _input a true
		if(!_input)		// Se è un neurone di _input, non ha sinapsi in ingresso
		{
			for(uint iS = 0; iS<get_n_syn(); iS++)
			{
				synapse &s = _syns[iS];
				s.set_node_ptr(&_net.get_neuron(ilay - 1, s._in));
			}
		}
	}



}