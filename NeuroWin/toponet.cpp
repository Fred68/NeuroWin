
#include "neuro_def.h"
#include "network.h"

namespace neuro
{
	toponet::toponet()
	{
		v_indx.clear();
	}

	toponet::~toponet()
	{
		v_indx.clear();
	}

	void toponet::update_topo(network &net)
	{
		bool ok = false;
		net.calc_indexes();									// Calcola gli indici di neuroni e sinapsi, prima del salvataggio
		v_indx.clear();
		v_indx.resize(net.get_n_layers());					// Vettore con i livelli
		for(uint iL = 0; iL<net.get_n_layers(); iL++)		// Percorre i livelli
		{
			layer &lay = net.get_layer(iL);
			v_indx[iL].resize(lay.size());					// Vettore con i nodi (un vettore per livello)
			for(uint iN = 0; iN < lay.size(); iN++)			// Percorre i nodi del livello
			{
				neuron &neu = net.get_neuron(iL,iN);
				v_indx[iL][iN].resize(neu.get_n_syn());		// Vettore con gli indici dei nodi (liv. prec.) delle sinapsi
				for(uint iS=0; iS < neu.get_n_syn(); iS++)
				{
					v_indx[iL][iN][iS] = neu.get_neuron_index(iS);
				}
			}
		}
	}

	uint toponet::get_layers_num()
	{
		uint ret = v_indx.size();
		return ret;
	}

	uint toponet::get_neurons_num(uint lay, bool &ok)
	{
		uint ret = 0;
		if(lay < v_indx.size())
		{
			ret = v_indx[lay].size();
		}
		else
		{
			ok = false;
		}
		return ret;
	}

	uint toponet::get_synapses_num(uint lay, uint n, bool &ok)
	{
		uint ret = 0;
		uint nn = get_neurons_num(lay, ok);
		if( ok && (n < nn))
		{
			ret = v_indx[lay][n].size();
		}
		else
		{
			ok = false;
		}
		return ret;
	}

	uint toponet::get_neuron_index_of_synapse(uint lay, uint n, uint s, bool &ok)
	{
		uint ret = 0;
		uint ns = get_synapses_num(lay,n,ok);
		if ( ok && (s < ns))
		{
			ret = v_indx[lay][n][s];
		}
		else
		{
			ok = false;
		}
		return ret;
	}

	bool toponet::isOk()
	{	
		bool ok = true;													// Cicli vari finché ok è true
		uint nl = get_layers_num();
		if(nl > 1)														// Almeno due livelli
		{
			for(uint il = 0; ok && (il < nl); il++)
			{
				uint nn = get_neurons_num(il,ok);
				if( ok && (nn > 0))										// Almeno un neurone
				{
					for(uint in = 0; ok && (in < nn); in++)
					{
						uint ns = get_synapses_num(il,in,ok);
						if(ok)											// Ammessa nessuna sinapsi	
						{
							for(uint is = 0; ok && (is < ns); is++)
							{
								uint nindx = get_neuron_index_of_synapse(il,in,is,ok);	// Indice del neurone del livello precedente
								if(ok && (il > 0))
								{
									if(!(nindx < get_neurons_num(il-1,ok)))	// Se l'indice è maggiore del numero di nodi
									{
										ok = false;
										break;
									}
								}
								else
								{
									ok = false;
									break;
								}

							}
						}
						else
						{
							ok = false;
							break;

						}
					}
				}
				else
				{
					ok = false;
					break;
				}

			}
		}
		else
		{	
			ok = false;
		}
		return ok;
	}




}
