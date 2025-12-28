

#include "neuro_def.h"
#include "network.h"

#include <sstream>
#include <numeric>		// std::accumulate

//#define _PARALLEL	true

namespace neuro
{
    /*******************************************/
    /*                                         */
    /* network                                 */
    /*                                         */
    /*******************************************/

    network::network(init_data &ini_data) /*: _pp(this)*/
    {
        if(!ini_data.is_ok())
		{
			throw std::exception("Initialization data invalid.");
			return;
		}
		


		_nLays = ini_data.get_layers_num();

		/*#ifdef ACT_DBL
			_err_tot = 0.0;
		#else
			_err_tot = 0.0f;
        #endif*/

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

				uint jmax = (uint) _layers.back().size();
				for(uint j=0; j<jmax; j++)
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
		//txt += std::format("E: {0}\n", _err_tot);

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

	void network::add_exception(neuro_exception &ex)
	{
		_exceptions.push_back(ex);
	}

	void network::clear_exceptions()
	{
		_exceptions.clear();
	}
	
	bool network::isOk()
	{
		uint count = std::count_if(_exceptions.begin(), _exceptions.end(), neuro_exception::is_ex_error);
		return (count == 0);
	}
	
	std::string network::get_exceptions_string(bool show_warnings)
	{
		std::string ret, txt, txt_err, txt_warn;
		uint count, count_warn, count_err;
		bool err, warn;

		auto func_sel = [&](neuro_exception &ex)
		{
			if( (ex.is_error() && err) || (!ex.is_error() && warn))
			{
				txt += "\n" + ex.what();
				count++;
			}
		};

		count_err = count_warn = 0;

		txt = "", err = true, warn = false, count = 0;
		std::for_each(std::execution::seq, _exceptions.begin(), _exceptions.end(), func_sel);
		if(count > 0)		txt_err = txt;
		count_err = count;

		if(show_warnings)
		{
			txt = "", err = false, warn = true, count = 0;
			std::for_each(std::execution::seq, _exceptions.begin(), _exceptions.end(), func_sel);
			if (count > 0)	txt_warn = txt;
			count_warn = count;
		}

		if(count_err == 0)
		{
			ret += "network is ok";
		}
		else
		{
			ret += std::format("network has {0} errors:",count_err);
			ret += txt_err;
		}

		if (count_warn > 0)
		{
			ret += std::format("\nnetwork has {0} warnings:", count_warn);
			ret += txt_warn;
		}
		return ret;
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
			auto v = std::ranges::iota_view((uint)0, (uint)inp_lay.size());
			std::atomic<bool> ok = true;
			auto func_set = [&](uint i) {ok = ok && get_at(0, i).set_x(inp_lay[i]); };
			std::for_each(get_exe_pol(EXE_POL::layer),v.begin(),v.end(),func_set);
			ret = ok;
		}
		return ret;
	}

	bool network::set_outputs(const std::vector<act> &out_lay, act &error_tot)
	{
		bool ret = false;
		if (out_lay.size() == _layers[_nLays-1].size() - 1)		// Ultimo livello
		{
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
			error_tot = errtot;
			ret = ok;
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

	bool network::prop_bw(const std::vector<act> &out_lay, act &error_tot)
	{	
		bool ok = false;
		try
		{	
			ok = set_outputs(out_lay, error_tot);
			if(ok)
			{
				for(uint lay = _nLays-1; lay > 0; lay--)		// Calcolo necessariamente sequenziale
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


	bool network::backward_propagate_no_check(const std::vector<act> &inp_lay, const std::vector<act> &out_lay, uint cycles, act &error_tot)
	{
		bool ok = true;
		for(uint i = 0; (i < cycles) && ok; i++)
		{
			ok = prop_fw(inp_lay);
			if(ok)
			{
				ok = prop_bw(out_lay,error_tot);
				update_w();
			}
		}
		return ok;
	}

	bool network::backward_propagate(const std::vector<act> &inp_lay, const std::vector<act> &out_lay, uint cycles, act &error_tot, std::chrono::milliseconds &msec_elap)
	{
		
		auto inizio = std::chrono::high_resolution_clock::now();	// std::chrono::steady_clock::time_point
		bool ok = true;
		if ((out_lay.size() == _layers[_nLays - 1].size() - 1) && (inp_lay.size() == _layers[0].size() - 1))
		{
			ok = backward_propagate_no_check(inp_lay, out_lay, cycles, error_tot);
		}
		auto fine = std::chrono::high_resolution_clock::now();
		msec_elap = std::chrono::duration_cast<std::chrono::milliseconds> (fine - inizio);

		return ok;
	}

	bool network::backward_propagate(std::shared_ptr<learn_data> pldata, const uint cycles, const uint subcycles, act &error_med, std::chrono::milliseconds &msec_elap)
	{
		auto inizio = std::chrono::high_resolution_clock::now();
		bool ok = true;
		std::vector<act> get_string_exceptions(pldata->get_data_size());	// Vettore con gli errori totali per tutti i casi

		if( (pldata->check_data_size()) && (get_string_exceptions.size() > 0))
		{
			for(uint ic=0; ic < cycles && ok; ic++)			// Ripete per il numero di cicli di apprendimento
			{
				uint idat = 0;
				for (auto it = pldata->begin(); it != pldata->end(); it++)	// Percorre i dati di apprendimento
				{
					const std::vector<act> vi = it.get_input_v();			// Coppia di vettori con i dati di ingresso...		
					const std::vector<act> vo = it.get_output_v();			// ...e di uscita desiderati
					if(!backward_propagate_no_check(vi, vo, subcycles, get_string_exceptions[idat]))	// Ripete bkprop e calcola errore tot
					{
						ok = false;
						std::cerr << "Error in back_propagate()" << std::endl;
						break;
					}
					idat++;
				}
				error_med = std::accumulate(get_string_exceptions.begin(),get_string_exceptions.end(),(act)0.0) / get_string_exceptions.size();
			}
		}
		else
		{
			ok = false;
			std::cerr << "Error in learn data size" << std::endl;
		}
		auto fine = std::chrono::high_resolution_clock::now();
		msec_elap = std::chrono::duration_cast<std::chrono::milliseconds>(fine - inizio);

		return ok;
	}

}