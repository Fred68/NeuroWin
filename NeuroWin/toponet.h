#ifndef TOPONET_H
#define TOPONET_H

#include "neuro_def.h"
#include "network.h"

#include <vector>


namespace neuro
{
	class toponet;
	class network;
	

	class toponet
	{
		std::vector<std::vector<std::vector<uint>>> v_indx;

		public:
			toponet();
			~toponet();

			/// <summary>
			/// Cancella e rigerena la topologia, leggendola da 'net'
			/// </summary>
			/// <param name="net"></param>
			/// <returns></returns>
			void update_topo(network &net);
			
			inline void clear() {v_indx.clear();}

			/// <summary>
			/// Numero di livelli
			/// </summary>
			/// <returns></returns>
			uint get_layers_num();
			/// <summary>
			///  Numero di neuroni nel livello lay
			/// Se errore, forza ok a false e restituisce zero
			/// </summary>
			/// <param name="lay"></param>
			/// <returns></returns>
			uint get_neurons_num(uint lay, bool &ok);
			/// <summary>
			/// Numero di sinapsi nel neurone n del livello lay
			/// Se errore, forza ok a false e restituisce zero
			/// </summary>
			/// <param name="lay"></param>
			/// <param name="n"></param>
			/// <returns></returns>
			uint get_synapses_num(uint lay, uint n, bool &ok);
			/// <summary>
			/// Indice del neurone a cui è connessa la sinapsi s
			/// del neurone n del livello lay
			/// Se errore, forza ok a false e restituisce zero
			/// </summary>
			/// <param name="lay"></param>
			/// <param name="n"></param>
			/// <param name="s"></param>
			/// <returns></returns>
			uint get_neuron_index_of_synapse(uint lay, uint n, uint s, bool &ok);

			bool isOk();

	};
}

#endif

