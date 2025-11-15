

#include "neuro.h"

namespace neuro
{
    /*******************************************/
    /*                                         */
    /* neuron                                  */
    /*                                         */
    /*******************************************/

    neuron::neuron()
    {
        x = y = 0;
		b = 0;
        set_fact(fact_default());
        active = true;
        input = false;
        #if _DEBUG_NEURO_DET
        cout << "neuron()\n";
        #endif
    }
	neuron::neuron(bool isInput) : neuron()
	{
		if(isInput)					// Se è un neurone di input, imposta la funzione di attivazion identità  (ed il flag)
		{
			set_fact(FACT::id);		// Prima imposta FACT...
			input = true;			// ...poi imposta input a true, che disabilita set_fact()
		}
	}
	neuron::neuron(std::vector<neuron> &prev, act std_w, act bias_w) : neuron()
    {
		for(uint i=0; i<prev.size(); i++)						// Imposta il vettore delle sinapsi (non è un neurone di input)
		{
			neuron &n = prev[i];
			syns.push_back(synapse(n, (i == prev.size() - 1) ? bias_w : std_w));
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
		// TODO mettere il numero di cifre in una costante
        std::string statStr = "";
        if(!active) statStr = "X";
		if(input)  statStr += "I";
		if(!statStr.empty())	statStr = "["+statStr+"]";
        std::string txt = format("x={0:.3f},y={1:.3f},b={4:.3f}(f={2}){3}",x,y,get_fact_name(),statStr,b);
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
                    txt = txt + std::format("[{0}{1:.3f}]", nn, s.w);
                }
            }
        }
        #if TXT_INFO
        txt = name + ": " + txt;
        #endif
        return txt;
    }    
 
	void neuron::set_active(bool stat) { active = stat; }
	void neuron::set_input(bool inp)
	{
		input = inp;
	}

	void neuron::set_fact(FACT f)
	{
		if(!input)			// SE è un neurone di input, la funzione di attivazione è quella definita nel costrutture
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
		return network::fact2string(fact);
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
	void neuron::set_b(act b_in) { b = b_in;}

    void neuron::calc_x()
    {
        if(active && !input)
        {
            #ifdef ACT_DBL
                act s0 = 0.0;
            #else
                act s0 = 0.0f;
            #endif
        
            // Calcola, su tutte le sinapsi del nodo, la somma delle uscite y dei nodi collegati, moltiplicate...
            // ...per il peso w della sinapsi. Il risultato è il segnale di ingresso x del nodo.        
            std::atomic<act> sum;
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
    FACT fact_default();



}