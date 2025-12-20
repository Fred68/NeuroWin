

#ifndef LEARN_DATA_H
#define LEARN_DATA_H

#include "neuro_def.h"
#include "network.h"

namespace neuro
{
	class network;

	class learn_data
	{
		
		public:
			static const uint UINT_ERROR = UINT_MAX;

		private:
			const std::shared_ptr<network> _pnet;			/// Puntatore alla rete
			const uint _inp_sz,_out_sz;						/// Lunghezze dei vettori di input e output richiesti dalla rete

			std::vector<std::vector<act>> _vinp;			/// Vettori di input
			std::vector<std::vector<act>> _vout;			/// Vettori di output
			std::vector<std::tuple<uint,uint>> _ldata;		/// Vettori dei valori per l'apprendimento


		public:

			learn_data(std::shared_ptr<network> pnet);
			std::string to_string();

			uint add_input(std::vector<act> v);
			uint add_output(std::vector<act> v);
			void add_data(uint index_input, uint index_output);
			void clear_data() {_ldata.clear();}
			void clear_all() {clear_data(); _vinp.clear(); _vout.clear();}

			std::vector<act> &get_input(uint i);
			std::vector<act> &get_output(uint i);
			std::tuple<std::vector<act>&, std::vector<act>&> get_data(uint i);
	};


}
#endif

