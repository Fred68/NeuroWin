

#ifndef LEARN_DATA_H
#define LEARN_DATA_H

#include "neuro_def.h"
#include "network.h"
#include <list>

namespace neuro
{
	class learn_data
	{

		std::vector<std::vector<act>> _vinp;			/// Valori di input size()==layer[0].size()
		std::vector<std::vector<act>> _vout;			/// Valori di output size()==layer[nLays-1].size()
		std::list<std::tuple<uint,uint>> _ldata;		/// Lista dei valori per l'apprendimento


		public:
			learn_data(uint inp_sz, uint out_sz);
			learn_data(const network &net);
	};

}
#endif

