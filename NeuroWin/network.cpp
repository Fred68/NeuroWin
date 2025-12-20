

#include "neuro_def.h"
#include "network.h"
#include <sstream>

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
        if(!ini_data.is_ok())
		{
			throw std::exception("Initialization data invalid.");
			return;
		}

		_nLays = ini_data.get_layers_num();

		#ifdef ACT_DBL
			_err_tot = 0.0;
		#else
			_err_tot = 0.0f;
        #endif

		_learn_const = ini_data.get_learn_const();
		set_f_learn(lcf_costant_value);

		// TODO: Impostare exe_pol in base al numero totale di nodi e al numero di nodi per livello
		// TODO: Fare delle prove di velocità
		exe_pol[(int)EXE_POL::neuron] = std::execution::par;
		exe_pol[(int)EXE_POL::layer] = std::execution::par;
		exe_pol[(int)EXE_POL::network] = std::execution::par;

		if(_nLays > 1)
		{
			for(uint i=0; i < _nLays; i++)		// Crea i livelli, ognuno con un nodo in più (uscita 1, disattivo, per i bias)
			{
				layer *ln;

				if(i==0)								// Per il primo livello, crea neuroni di input, usando come ctor:
				{										// ...neuron(bool true) 
					ln = new layer(ini_data.get_layers()[i] + 1, *this,true );
				}
				else
				{										// Per gli altri livelli, usa neuron()   
					ln = new layer(ini_data.get_layers()[i] + 1, *this, _layers[i - 1]);
				}
				_layers.push_back(*ln);

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
						_layers.back()[j].set_fact(ini_data.get_types()[i]);
					}
				}
			}

			_nInputs = _layers[0].size()-1;
			_nOutputs = _layers[_nLays-1].size()-1;
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
		txt += std::format("Learn const: {0}\n", get_learn_const());
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

	std::string network::display_vector(std::vector<act> &v)
	{
		std::ostringstream ss;
		char sep = '\t';
		ss << "[";
		for(uint i=0; i < v.size(); i++)
		{
			sep = (i < v.size() - 1) ? '\t' : '\0';
			ss << std::format(TO_STR_FORMAT_FLOAT(3),v[i]) << sep;
		}
		ss << "]";
		return ss.str();
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

	bool network::set_inputs(const std::vector<act> &inp_lay)
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
		//std::vector<neuron> &lay = _layers[nlay];
		layer &lay = _layers[nlay];
		auto v = std::ranges::iota_view((uint)0, (uint)lay.size());
		auto func_calc_y = [&](uint i) {lay[i].calc_x(); lay[i].calc_y(); lay[i].set_beta((act)0.0);};
		std::for_each(get_exe_pol(EXE_POL::layer), v.begin(), v.end(), func_calc_y);	// Formula [2]		
		return ret;
	}

	bool network::calc_ei_eaprec_lay(uint nlay)
	{
		bool ret = false;
		//std::vector<neuron> &lay = _layers[nlay];
		layer &lay = _layers[nlay];
		auto v = std::ranges::iota_view((uint)0, (uint)lay.size());
		auto func_calc_ei = [&](uint j) {lay[j].calc_ei(); lay[j].calc_parz_eai(); };
		std::for_each(get_exe_pol(EXE_POL::layer), v.begin(), v.end(), func_calc_ei);
		return ret;
	}

	void network::calc_w_lay(uint nlay)
	{
		layer &lay = _layers[nlay];
		if(lay.get_recalc_w())
		{
			auto v = std::ranges::iota_view((uint)0, (uint)lay.size());
			auto func_calc_w = [&](uint i) {lay[i].calc_w(get_f_learn()(*this,nlay,i));};
			// Nota: calc_w(...) usa std::execution::par sui pesi, possibile 'race condition' ?
			// No, le sinapsi di un nodo (verso i precedenti) sono indipendenti da quelle di un altro nodo
			std::for_each(get_exe_pol(EXE_POL::layer), v.begin(), v.end(), func_calc_w);
		}
	}


	bool network::prop_fw(const std::vector<act> &inp_lay)
	{
		bool ok = set_inputs(inp_lay);
		if(ok)
		{
			for(uint i = 0; i<_nLays; i++)		// Calcolo necessariamente sequenziale
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
			if(ok)
			{
				for(int lay = _nLays-1; lay > 0; lay--)		// Calcolo necessariamente sequenziale
				{
					calc_ei_eaprec_lay(lay);
				}
			}
		}
		catch(std::exception const &ex)
		{
			std::cerr << ex.what() << std::endl;
		}
		return ok;
	}

	void network::update_w()
	{
		auto v = std::ranges::iota_view((uint)1, _nLays);	// Dal secondo livello (lay=1) all'ultimo (nLay-1).
		auto func_calc_lay = [&](uint i) {calc_w_lay(i);};
		std::for_each(get_exe_pol(EXE_POL::network),v.begin(),v.end(),func_calc_lay);
	}


	bool network::forward_propagate(const std::vector<act> &inp_lay, std::vector<act> &out_lay)
	{
		bool ok;
		if(out_lay.size() == _layers[_nLays - 1].size() - 1)
		{
			if(ok = prop_fw(inp_lay))
			{
				for(int i=0; i< out_lay.size(); i++)
				{
					out_lay[i] = get_neuron(_nLays - 1,i).get_y();
				}
			}
		}
		else
		{
			ok = false;
		}
		return ok;
	}

	bool network::backward_propagate(const std::vector<act> &inp_lay, std::vector<act> &out_lay, uint cycles, std::chrono::milliseconds &msec_elap)
	{
		bool ok = true;
		auto inizio = std::chrono::high_resolution_clock::now();	// std::chrono::steady_clock::time_point

		if ( (out_lay.size() == _layers[_nLays - 1].size() - 1) && (inp_lay.size() == _layers[0].size()-1))
		{
			for(uint i = 0; (i < cycles) && ok; i++)
			{
				ok = prop_fw(inp_lay);
				if(ok)
				{
					ok = prop_bw(out_lay);
					update_w();
				}
			}
		}
		else
		{
			ok = false;
		}
		auto fine = std::chrono::high_resolution_clock::now();

		msec_elap = std::chrono::duration_cast<std::chrono::milliseconds> (fine - inizio);
		
		return ok;
	}


}