
#include "neuron_synapse.h"
#include "network.h"		// Nesessario, se no tipo incompleto "neuro::network" non consentito

namespace neuro
{
    /*******************************************/
    /*                                         */
    /* neuron                                  */
    /*                                         */
    /*******************************************/

    neuron::neuron(const network &netwrk) : net(netwrk)								//neuron::neuron(std::shared_ptr<network> netwrk) : pnet(netwrk)
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
	neuron::neuron(const network &netwrk, bool isInput) : neuron(netwrk)			//neuron::neuron(std::shared_ptr<network> netwrk, bool isInput) : neuron(netwrk)
	{
		if(isInput)					// Se è un neurone di input, imposta la funzione di attivazion identità  (ed il flag)
		{
			set_fact(FACT::id);		// Prima imposta FACT...
			input = true;			// ...poi imposta input a true, che disabilita set_fact()
		}
	}
	neuron::neuron(const network &netwrk, std::vector<neuron> &prev, act neu_w, act bias_w) : neuron(netwrk)		//neuron::neuron(std::shared_ptr<network> netwrk, std::vector<neuron> &prev, act neu_w, act bias_w) : neuron(netwrk)
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
		std::string type_err_der = "b_ei";
        if(!active) statStr = "X";
		if(input)  statStr += "I";
		#if _DEBUG
		if(isBeta)
			type_err_der = "beta";
		else
			type_err_der = "ei";
		#else

		#endif
		if(!statStr.empty())	statStr = "["+statStr+"]";

		std::string txt = format(to_string_frm_n, x, y, get_fact_name(), statStr, ei, type_err_der);
        if(active)
        {
            for(synapse s : syns)
            {
                if(s.pn != nullptr)
                {
                    std::string nn = "";
                    #if TXT_INFO
                    nn = s.pn->get_name()+",";
                    #endif
                    txt = txt + std::format(to_string_frm_w, nn, s.w);
                }
            }
        }
        #if TXT_INFO
        txt = name + ": " + txt;
        #endif
		//net.get_layers_count();
        return txt;
    }    
 
	void neuron::set_active(bool stat) { active = stat; }
	void neuron::set_input(bool inp)
	{
		input = inp;
	}

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
				throw std::exception("enum FACT non riconosciuto.");
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
		#if _DEBUG
		isBeta = false;
		#endif
	}
	void neuron::set_beta(act beta_in)
	{
		beta = beta_in;
		#if _DEBUG
		isBeta = true;
		#endif
	}
	act neuron::get_ei()
	{	
		#if _DEBUG
		if(isBeta)
			throw std::exception("Get EI when beta is set.");
		#endif
		return ei;
		
	}
	act neuron::get_beta()
	{
		#if _DEBUG
		if(!isBeta)
			throw std::exception("Get beta when EI is set.");
		#endif
		return beta;
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
            auto func_add = [&](const synapse &s) {sum.fetch_add(s.pn->y * s.w);};
            std::for_each(std::execution::par, syns.begin(), syns.end(), func_add);
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
			#if _DEBUG
			if(!isBeta)
				throw std::exception("Cannot calc. EI when beta is not set");
			#endif
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
			auto func_ea = [&](const synapse &s) {s.pn->set_beta(s.w * get_ei());};
			std::for_each(std::execution::par, syns.begin(), syns.end(), func_ea);
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
				s.w += - learn_const * ei * s.pn->y;
			};
			std::for_each(std::execution::par, syns.begin(), syns.end(), func_updw);
		}
		uint y = net.test();
	}

	/*******************************************/
    // Funzioni di attivazione
    act neuron::sigmoid(neuron *n)
    {
        #ifdef ACT_DBL
            return 1.0 / (1.0 + std::exp(n->x));
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



}