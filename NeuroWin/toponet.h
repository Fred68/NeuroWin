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

			uint get_layers_num();
			uint get_neurons_num(uint lay);
			uint get_synapses_num(uint lay, uint n);

			bool isOk();

	};
}

#endif

