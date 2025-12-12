

#include "neuro_def.h"
#include "network.h"

#define _PARALLEL	true

namespace neuro
{
    /*******************************************/
    /*                                         */
    /* network                                 */
    /*                                         */
    /*******************************************/

    network::network(init_data &ini_data)
    {
        _nLays = std::min( (uint) ini_data._layers.size(), (uint)ini_data._types.size() );

		if(ini_data._layers.size() != ini_data._types.size())
		{
			_nLays = 0;
			throw std::exception("layers and types size don't match.");
			return;
		}

		#ifdef ACT_DBL
			_err_tot = 0.0;
		#else
			_err_tot = 0.0f;
        #endif

		_learn_const = ini_data._learn_const;
		set_f_learn(lcf_costant_value);

		// TODO: impostare exe_pol in base al numero totale di nodi e al numero di nodi per livello
		// TODO: fare delle prove di velocità
		exe_pol[(int)EXE_POL::neuron] = std::execution::par;
		exe_pol[(int)EXE_POL::layer] = std::execution::par;
		exe_pol[(int)EXE_POL::network] = std::execution::par;

		if(_nLays > 1)
		{
			for(uint i=0; i < _nLays; i++)		// Crea i livelli, ognuno con un nodo in più (uscita 1, disattivo, per i bias)
			{
				std::vector<neuron> *vn;				// Alloca il vettore con vector<T>(numero, parametri per il ctor di T).
				if(i==0)								// Per il primo livello, crea neuroni di input, usando come ctor:
				{										// ...neuron(bool true) 
					vn = new std::vector<neuron>(ini_data._layers[i] + 1,{*this,true});
				}
				else
				{										// Per gli altri livelli, usa neuron()   
					vn = new std::vector<neuron>(ini_data._layers[i] + 1, {*this,_layers[i-1]});
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
					//_layers.back()[j].set_learn(static_learn_default);
					//_layers.back()[j].set_learn(_f_learn);
				}
			}
		}
		else
		{
			throw std::exception("Minimum 2 layer required.");
			return;
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

    std::string network::to_string()
    {
        std::string txt;
        txt += std::format("Layers: {0}\n", _nLays);
		txt += std::format("E: {0}\n", _err_tot);

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
            throw std::exception("Layer out of range");
        else if (num >= _layers[lay].size())
            throw std::exception("Node out of range");
        else
            return get_at(lay, num);
    }

    #if TXT_INFO
    void network::name_elements()
    {
        for (uint i = 0; i < _nLays; i++)
            for (int j = 0; j < _layers[i].size(); j++)
            {
                get_at(i,j).set_name(std::format("L{0}N{1}",i,j));
            }
    }
    #endif

	bool network::set_inputs(std::vector<act> &inp_lay)
	{
		bool ret = false;
		if(inp_lay.size() == _layers[0].size()-1)		// 1° livello
		{
			#if !_PARALLEL
			bool ok = true;
			for(uint i=0; i<inp_lay.size(); i++)
			{
				ok = ok && get_at(0,i).set_x(inp_lay[i]);
			}
			ret = ok;
			#else
			// TODO Fare prove di velocità. La versione parallela con iota potrebbe essere complessivamente più lenta. 
			auto v = std::ranges::iota_view((uint)0, (uint)inp_lay.size());
			std::atomic<bool> ok = true;
			auto func_set = [&](uint i) {ok = ok && get_at(0, i).set_x(inp_lay[i]); };
			std::for_each(get_exe_pol(EXE_POL::layer),v.begin(),v.end(),func_set);
			ret = ok;
			#endif
		}
		return ret;
	}

	bool network::set_outputs(std::vector<act> &out_lay)
	{
		bool ret = false;
		if (out_lay.size() == _layers[_nLays-1].size() - 1)		// Ultimo livello
		{
			#if !_PARALLEL
			for(uint i=0; i<out_lay.size(); i++)
			{
				get_at(_nLays - 1, i).set_beta(get_at(_nLays - 1, i).get_y() - out_lay[i]);
			}
			#else
			// TODO Fare prove di velocità. E' possibile che la versione parallela con iota sia complessivamente più lenta.
			auto v = std::ranges::iota_view((uint)0, (uint)out_lay.size());
			std::atomic<bool> ok = true;
			#ifdef ACT_DBL
                act s0 = 0.0;
            #else
                act s0 = 0.0f;
            #endif
			std::atomic<act> errtot = s0;
			auto func_set =	[&](uint i)
				{
					//get_at(_nLays - 1, i).set_beta( get_at(_nLays - 1, i).get_y()-out_lay[i]);
					auto betatmp = get_at(_nLays - 1, i).get_y() - out_lay[i];
					get_at(_nLays - 1, i).set_beta(betatmp);
					errtot.fetch_add(betatmp*betatmp);
				}; 
			std::for_each(get_exe_pol(EXE_POL::layer),v.begin(),v.end(),func_set);			// Formula [6]		
			ret = ok;
			_err_tot = errtot;

			#endif
			ret = true;
		}
		else
			throw std::exception("Output vector and last layer sizes don't match");
		return ret;
	}

	void network::set_weights(weight_func wf)
	{
		for(uint il = 1; il < _nLays; il++)
		{
			for(uint in = 0; in < _layers[il].size(); in++)
			{
				neuron n = get_at(il,in);						// Reference al neurone
				for(uint is = 0; is < n.get_n_syn(); is++)
				{
					bool is_bias = (is == (n.get_n_syn() - 1));	// Ultima sinapsi connessa a nodo One disattivo: il peso è il bias.
					n.set_w(wf(il, in, is, is_bias), is);		// Usa il puntatore a funzione
				}
			}
		}
	}

	act network::set_w_const(uint iLay, uint iNeu, uint iSyn, bool is_bias)
	{
		return is_bias ? (act)neuron::b_ini_const : (act)neuron::w_ini_const;
	};
	
	act network::set_w_mean(uint iLay, uint iNeu, uint iSyn, bool is_bias)
	{
		if(is_bias)
		{
			return (act)neuron::b_ini_mean;
		}
		else
		{
			uint nn = get_at(iLay,iNeu).get_n_syn();
			return (act) (neuron::w_ini_mean/nn);
		}

	}


	bool network::calc_y_lay(uint nlay)
	{
		bool ret = true;
		std::vector<neuron> &layer = _layers[nlay];
		auto v = std::ranges::iota_view((uint)0, (uint)layer.size());
		auto func_calc_y = [&](uint i) {layer[i].calc_x(); layer[i].calc_y(); layer[i].set_beta((act)0.0);};
		std::for_each(get_exe_pol(EXE_POL::layer), v.begin(), v.end(), func_calc_y);	// Formula [2]		
		return ret;
	}

	bool network::calc_ei_eaprec_lay(uint nlay)
	{
		bool ret = false;
		std::vector<neuron> &layer = _layers[nlay];
		auto v = std::ranges::iota_view((uint)0, (uint)layer.size());
		auto func_calc_ei = [&](uint j) {layer[j].calc_ei(); layer[j].calc_parz_eai(); };
		std::for_each(get_exe_pol(EXE_POL::layer), v.begin(), v.end(), func_calc_ei);
		return ret;
	}

	bool network::calc_w_lay(uint nlay)
	{
		bool ret = true;
		std::vector<neuron> &layer = _layers[nlay];
		auto v = std::ranges::iota_view((uint)0, (uint)layer.size());
		auto func_calc_w = [&](uint i) {layer[i].calc_w(get_f_learn()(*this,nlay,i));};
		// Nota: calc_w(...) usa std::execution::par sui pesi, possibile 'race condition' ?
		// No, le sinapsi di un nodo (verso i precedenti) sono indipendenti da quelle di un altro nodo
		std::for_each(get_exe_pol(EXE_POL::layer), v.begin(), v.end(), func_calc_w);
		return ret;
	}


	bool network::prop_fw(std::vector<act> &inp_lay)
	{
		bool ok = set_inputs(inp_lay);
		if(ok)
		{
			for(uint i = 0; i<_nLays; i++)				// Ciclo (qui non usa il calcolo parallelo)
			{
				calc_y_lay(i);	
			}
		}
		return ok;
	}

	bool network::prop_bw(std::vector<act> &out_lay)
	{	
		bool ok = false;
		try
		{
			ok = set_outputs(out_lay);
			for(int lay = _nLays-1; lay > 0; lay--)
			{
				calc_ei_eaprec_lay(lay);
			}
		}
		catch(std::exception const &ex)
		{
			std::cout << ex.what() << std::endl;
		}
		return ok;
	}

	bool network::update_w()
	{
		bool ok = true;
		auto v = std::ranges::iota_view((uint)1, _nLays);	// Dal secondo livello (lay=1) all'ultimo (nLay-1).
		auto func_calc_lay = [&](uint i) {calc_w_lay(i);};
		std::for_each(get_exe_pol(EXE_POL::network),v.begin(),v.end(),func_calc_lay);
		return ok;
	}


	uint network::test() const
	{
		uint x = 1;
		return x;
	};


}