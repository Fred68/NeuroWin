

#include "neuro.h"

#define _PARALLEL	false

namespace neuro
{
    /*******************************************/
    /*                                         */
    /* network                                 */
    /*                                         */
    /*******************************************/

    network::network()
    {
        _nLays = 0;
        #if _DEBUG
        std::cout << "network()\n";
        #endif
    }

    network::network(init_data &ini_data)
    {
        _nLays = std::min( (uint) ini_data._layers.size(), (uint)ini_data._types.size() );

        for(uint i=0; i < _nLays; i++)		// Crea i livelli, ognuno con un nodo in più (uscita 1, per i bias)
        {
            std::vector<neuron> *vn;				// Alloca il vettore con vector<T>(numero, parametri per il ctor di T).
            if(i==0)								// Per il primo livello, crea neuroni di input, usando come ctor:
            {										// ...neuron(bool true) 
                vn = new std::vector<neuron>(ini_data._layers[i] + 1,true);    
            }
            else
            {										// Per gli altri, usa neuron()   
                vn = new std::vector<neuron>(ini_data._layers[i] + 1,_layers[i-1]);   // allocate vector<neuron>
            }
            _layers.push_back(*vn);

            int jmax = (uint) _layers.back().size();
            for(int j=0; j<jmax; j++)
            {
                if(j == jmax-1)
                {
                    _layers.back()[j].set_fact(FACT::one);           // Nodo aggiunto: uscita sempre a 1
                    _layers.back()[j].calc_y();                      // Calcola l'uscita e...
                    _layers.back()[j].set_active(false);             // ...disattiva
                }
                else
                {
                    _layers.back()[j].set_fact(ini_data._types[i]);
                }
            }
        }
        
        #if TXT_INFO
        name_elements();
        #endif
        
        #if _DEBUG
        std::cout << "network(" << _nLays <<")\n";
        #endif

       
    }
    network::~network()
    {
        _nLays = 0;

        #if _DEBUG
        std::cout << "~network()\n";
        int x = getchar();
        #endif
    }

    std::string network::fact2string(FACT f)
    {
        std::string str = "";
        switch(f)
        {
            case FACT::one:
                str = "one";
                break;
            case FACT::sigmoid:
                str = "sigmoid";
                break;
            case FACT::tanh:
                str = "tanh";
                break;
            case FACT::relu:
                str = "relu";
                break;
            case FACT::id:
                str = "id";
                break;
            default:
                str = "FACT error";
                break;
        }
        return str;
    }
    std::string network::to_string()
    {
        std::string txt;
        txt += std::format("Layers: {0}\n", _nLays);

        for (uint i=0; i < _nLays; i++)
        {
            txt += std::format("Layer: {0}\n", i);
            for(neuron n : _layers[i])
            {
                txt += format("{0}\n", n.to_string());
            }
        }
        return txt;
    }
    neuron& network::get_neuron(uint lay, uint num)
    {
        if (lay >= _nLays)
            throw new std::exception("Layer out of range");
        else if (num >= _layers[lay].size())
            throw new std::exception("Node out of range");
        else
            return get_at(lay, num);
    }

    #if TXT_INFO
    void network::name_elements()
    {
        for (int i = 0; i < _nLays; i++)
            for (int j = 0; j < layers[i].size(); j++)
            {
                get_at(i,j).set_name(std::format("L{0}N{1}",i,j));
            }
    }
    #endif

	bool network::set_input_layer(std::vector<act> &inp_lay)
	{
		bool ret = false;
		if(inp_lay.size() == _layers[0].size()-1)
		{
			#if !_PARALLEL
			bool ok = true;
			for(uint i=0; i<inp_lay.size(); i++)
			{
				ok = ok && get_at(0,i).set_x(inp_lay[i]);
			}
			ret = ok;
			#else
			// E' possibile che la versione parallela con iota sia complessivamente più lenta.
			// TODO Fare prove di velocità
			auto v = std::ranges::iota_view((uint)0, (uint)inp_lay.size());
			std::atomic<bool> ok = true;
			auto func_set = [&](uint i) {ok = ok && get_at(0, i).set_x(inp_lay[i]); };
			std::for_each(std::execution::par,v.begin(),v.end(),func_set);
			ret = ok;
			#endif
		}
		return ret;
	}

	bool network::calc_lay(uint nlay)
	{
		bool ret = false;
		std::vector<neuron> &layer = _layers[nlay];							// Riferimento al livello da calcolare
		auto v = std::ranges::iota_view((uint)0, (uint)layer.size());		// 0, 1, 2... per calc. parallelo
		//std::atomic<bool> ok = true;
		auto func_calc = [&](uint i) {layer[i].calc_x(); layer[i].calc_y();};
		std::for_each(std::execution::par, v.begin(), v.end(), func_calc);
		return ret;
	}

	bool network::fw_prop(std::vector<act> &inp_lay)
	{
		bool ok = set_input_layer(inp_lay);
		if(ok)
		{
			for(uint i = 0; i<_nLays; i++)				// Ciclo (qui non usa il calcolo parallelo)
			{
				calc_lay(i);	
			}
		}
		return ok;
	}



}