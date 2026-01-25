
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
		uint ret = UINT_ERROR;
		ret = v_indx.size();
		return ret;
	}

	uint toponet::get_neurons_num(uint lay)
	{
		uint ret = UINT_ERROR;
		if(lay < v_indx.size())
		{
			ret = v_indx[lay].size();
		}
		return ret;
	}

	uint toponet::get_synapses_num(uint lay, uint n)
	{
		uint ret = UINT_ERROR;
		uint nn = get_neurons_num(lay);
		if( (nn != UINT_ERROR) && (n < nn))
		{
			ret = v_indx[lay][n].size();
		}
		return ret;
	}

	bool toponet::isOk()
	{	
		bool ok = true;													// Cicli vari finché ok è true
		uint nl = get_layers_num();
		if((nl != UINT_ERROR) && (nl > 1))								// Almeno due livelli
		{
			for(uint il = 0; ok && (il < nl); il++)
			{
				uint nn = get_neurons_num(il);
				if( (nn != UINT_ERROR) && (nn > 0))						// Almeno un neurone
				{
					for(uint in = 0; ok && (in < nn); in++)
					{
						uint ns = get_synapses_num(il,in);
						if(ns != UINT_ERROR)							// Ammessa nessuna sinapsi	
						{
							for(uint is = 0; ok && (is < ns); is++)
							{
								uint nindx = v_indx[il][in][is];		// Indice del neurone del livello precedente
								if(il > 0)  
								{
									if(!(nindx < get_neurons_num(il-1)))	// Se l'indice è maggiore del numero di nodi
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
