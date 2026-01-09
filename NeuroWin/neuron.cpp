
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

    neuron::neuron(network &netwrk) : net{netwrk}
    {
        x = y = 0;
		ei = 0;
        set_fact(fact_default());
        active = true;
        input = false;
        #if _DEBUG_NEURO_DET
        cout << "neuron()\n";
        #endif
    }
	neuron::neuron(network &netwrk, bool isInput) : neuron{netwrk}
	{
		if(isInput)					// Se è un neurone di input, imposta la funzione di attivazion identità  (ed il flag)
		{
			set_fact(FACT::id);		// Prima imposta FACT...
			input = true;			// ...poi imposta input a true, che disabilita set_fact()
		}
	}
	neuron::neuron(network &netwrk, std::vector<neuron> &prev, act neu_w, act bias_w) : neuron(netwrk)
    {
		for(uint i=0; i<prev.size(); i++)						// Imposta il vettore delle sinapsi (non è un neurone di input)
		{														// con pesi e bias indicati
			neuron &n = prev[i];
			syns.push_back(synapse(n, (i == prev.size() - 1) ? bias_w : neu_w));
		}
        #if _DEBUG_NEURO_DET
        cout << "neuron(neuron &prev)\n";
        #endif
    }

	#if _COPY_CTORS_
	neuron::neuron(const neuron& other) :	net{other.net},
											x{other.x}, y{other.y},
											f_act{other.f_act}, f_act_der{other.f_act_der}, fact{other.fact},
											active{other.active}, input{other.input},
											_nstat{other._nstat}
											//syns(other.syns.size())
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
		for(uint i=0; i<other.syns.size(); i++)
		{
			//syns[i] = other.syns[i];
			#error AGGIORNARE IL RIFERIMENTO DELLA SINAPSI ALLA NUOVA RETE
			syns.push_back(synapse(other.syns[i]));
		}
	}
	neuron& neuron::operator=(const neuron& other)
	{
		net = other.net;
		x = other.x;
		y = other.y;
		f_act = other.f_act;
		f_act_der = other.f_act_der;
		fact = other.fact;
		active = other.active;
		input = other.input;
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
		syns.clear();
		//syns.resize(other.syns.size());
		for (uint i = 0; i < other.syns.size(); i++)
		{
			//syns[i] = other.syns[i];
			#error AGGIORNARE IL RIFERIMENTO DELLA SINAPSI ALLA NUOVA RETE
			syns.push_back(synapse(other.syns[i]));
		}
		return *this;
	}
	#endif
	#if _MOVE_CTORS_
	neuron::neuron(neuron&& other) :	net{ other.net },
										x{ other.x }, y{ other.y },
										f_act{ other.f_act }, f_act_der{ other.f_act_der }, fact{ other.fact },
										active{ other.active }, input{ other.input },
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
		syns.clear();
		syns = std::move(other.syns);
	}
	neuron& neuron::operator=(neuron&& other)
	{
		net = other.net;
		x = other.x;
		y = other.y;
		f_act = other.f_act;
		f_act_der = other.f_act_der;
		fact = other.fact;
		active = other.active;
		input = other.input;
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
		syns.clear();
		syns = std::move(other.syns);
		return *this;
	}
	#endif

    neuron::~neuron()
    {
		// vector<synapse> non ha bisogno di dtor.
        #if _DEBUG_NEURO_DET
        cout << "~neuron()\n";
        getchar();
        #endif
    }
    
    std::string neuron::to_string()
    {
        std::string statStr = "";
		std::string type_stat = "?";
        if(!active) statStr = "X";
		if(input)  statStr += "I";
	
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

        if(active)
        {
            for(synapse s : syns)
            {
                if(s._pn != nullptr)			// if (std::get<ptN>(s._pn) != nullptr)
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
 
	void neuron::set_active(bool stat)	{ active = stat; }
	void neuron::set_input(bool inp)	{ input = inp; }

	void neuron::set_fact(FACT f)
	{
		if(!input)			// Se è un neurone di input, la funzione di attivazione è quella definita nel costrutture
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
				f_act = &one;
				f_act_der = &zero;
				break;
			case FACT::id:
				f_act = &id;
				f_act_der = &one;
				break;
			default:
				throw net.create_exception(network::neuro_exception::type::activation_function, true, "in neuron::set_fact()");
				
			}
			fact = f;
		}
	}
	std::string neuron::get_fact_name()
	{
		return fact2string(fact);
	}
	
    bool neuron::set_x(act x_in)
    {
        if(input)
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
	void neuron::set_index(uint i)
	{
		index_in_layer = i;
		_nstat = stat::_index;
	}

	act neuron::get_ei()
	{	
		if(_nstat!=stat::_ei)
			throw net.create_exception(network::neuro_exception::type::EI_mismatch, true, "failed neuron::get_ei(), EI is not");
		return ei;
		
	}
	act neuron::get_beta()
	{
		if (_nstat != stat::_beta)
			throw net.create_exception(network::neuro_exception::type::beta_mismatch, true, "failed neuron::get_beta(), beta is not set");
		return beta;
	}
	uint neuron::get_index()
	{
		if (_nstat != stat::_index)
			throw net.create_exception(network::neuro_exception::type::beta_mismatch, true, "failed neuron::get_index(), index is not set");
		return index_in_layer;
	}

	act neuron::get_w(uint i)
	{
		return (i < syns.size()) ? syns[i].w : 0;
	}
	void neuron::set_w(act w, uint i)
	{
		if(i < syns.size())
			syns[i].w = w;
	}

    void neuron::calc_x()
    {
        if(active && !input)
        {
            #ifdef ACT_DBL
				std::atomic<act> sum = 0.0;
            #else
				std::atomic<act> sum = 0.0f;
            #endif
        
            // Calcola, su tutte le sinapsi del nodo, la somma delle uscite y dei nodi collegati, moltiplicate...
            // ...per il peso w della sinapsi. Il risultato è il segnale di ingresso x del nodo.        
            auto func_add = [&](const synapse &s) {sum.fetch_add(s._pn->y * s.w);};
			//auto func_add = [&](const synapse &s) {sum.fetch_add(std::get<ptN>(s._pn)->y * s.w); };
            std::for_each(net.get_exe_pol(EXE_POL::neuron), syns.begin(), syns.end(), func_add);
			x = sum;
        }
    }
    void neuron::calc_y()
    {   
        if(active)
            y = f_act(this);
    }
	void neuron::calc_ei()
	{
		if(active)
		{
			if (_nstat != stat::_beta)
				throw net.create_exception(network::neuro_exception::type::beta_mismatch, true, "failed neuron::calc_ei(), beta is not set");
			set_ei(get_beta() * f_act_der(this));
		}
		else
		{
			set_ei((act)0.0);
		}
	}
	void neuron::calc_parz_eai()
	{
		if (active && !input)
		{
			auto func_ea = [&](const synapse &s) {s._pn->set_beta(s.w * get_ei());};
			//auto func_ea = [&](const synapse &s) {std::get<ptN>(s._pn)->set_beta(s.w * get_ei()); };
			std::for_each(net.get_exe_pol(EXE_POL::neuron), syns.begin(), syns.end(), func_ea);
		}
	}
	void neuron::calc_w(act learn_const)
	{
		if (active && !input)
		{
			auto func_updw = [&](synapse &s)
			{	
				// Corregge il peso wi della sinapsi tra in nodo j attuale e il nodo i precedente
				// con le formule [8] e [10], usando il prodotto tra ei (del nodo j) e y (del nodo i).
				//s.w -=  learn_const * ei * std::get<ptN>(s._pn)->y;
				s.w -= learn_const * ei * s._pn->y;
			};
			std::for_each(net.get_exe_pol(EXE_POL::neuron), syns.begin(), syns.end(), func_updw);
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
				size_t ssz = syns.size();
				fs.write(reinterpret_cast<char*>(&index_in_layer),sizeof(index_in_layer));
				fs.write(reinterpret_cast<char*>(&fact), sizeof(fact));
				fs.write(reinterpret_cast<char*>(&active), sizeof(active));
				fs.write(reinterpret_cast<char*>(&input), sizeof(input));
				fs.write(reinterpret_cast<char*>(&ssz), sizeof(ssz));
				for(uint i=0; i<ssz; i++)
				{
					syns[i].write(fs);
				}
			}
			else
			{
				throw net.create_exception(network::neuro_exception::type::index_mismatch, true, "Neuron index is not set");
			}
		}
		catch(std::exception &ex)
		{
			std::cerr << "Eccezione exception in neuron::write(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		}
		catch (network::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in neuron::write(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		}
	}

	void neuron::read(std::ifstream &fs)
	{
		try
		{
			uint i_tmp;
			FACT f_tmp;
			bool active_tmp, input_tmp;
			size_t sz_tmp;

			fs.read(reinterpret_cast<char*>(&i_tmp), sizeof(i_tmp));
			fs.read(reinterpret_cast<char*>(&f_tmp), sizeof(f_tmp));
			fs.read(reinterpret_cast<char*>(&active_tmp), sizeof(active_tmp));
			fs.read(reinterpret_cast<char*>(&input_tmp), sizeof(input_tmp));
			fs.read(reinterpret_cast<char*>(&sz_tmp), sizeof(sz_tmp));

			index_in_layer = i_tmp;
			fact = f_tmp;
			active = active_tmp;
			input = input_tmp;

			syns.clear();
			syns.resize(sz_tmp);
			for (uint i = 0; i < sz_tmp; i++)
			{
				//synapse s;
				//s.read(fs);
				syns[i].read(fs);
				//syns.push_back(s);
			}

		}
		catch (std::exception &ex)
		{
			std::cerr << "Eccezione exception in neuron::read(...): " << ex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		} catch (network::neuro_exception &nex)
		{
			std::cerr << "Eccezione neuro_exception in neuron::read(...): " << nex.what() << std::endl;
			// TODO poi aggiungere (con o senza throw) net.create_exception...
		}
	}

	void neuron::update_ptr(uint in, uint ilay)
	{
		if(!input)		// Se è un neurone di input, non ha sinapsi in ingresso
		{
			for(uint iS = 0; iS<get_n_syn(); iS++)
			{
				synapse &s = syns[iS];
				s._pn = std::shared_ptr<neuron>(&(net.get_neuron(ilay-1,s._in)));
			}
		}
	}



}