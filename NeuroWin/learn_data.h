

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
			static const uint UINT_ERROR = UINT_MAX;		/// Errorw per uint, equivalente a (uint) -1;

		private:
			const std::shared_ptr<network> _pnet;			/// Puntatore alla rete
			const uint _inp_sz,_out_sz;						/// Lunghezze dei vettori di input e output richiesti dalla rete

			std::vector<std::vector<act>> _vinp;			/// Vettori di input
			std::vector<std::vector<act>> _vout;			/// Vettori di output
			std::vector<std::tuple<uint,uint>> _ldata;		/// Vettori dei valori per l'apprendimento
		
		public:
			/// <summary>
			/// Iteratore semplice con soltanto begin, end e ++
			/// Non implementato for(Iterator it : learn_data _ldata) perché restutuirebbe una tuple<> difficile da gestire
			/// </summary>
			class Iterator
			{
				using iterator_category = std::forward_iterator_tag;
				using difference_type = std::ptrdiff_t;
				using value_type = uint;
				using pointer = uint*;
			
				public:
					Iterator(uint i, learn_data &ld) : _indx(i), _ld(ld) {}
					Iterator& operator++() { _indx++; return *this; }
					Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }
					friend bool operator== (const Iterator& a, const Iterator& b) { return a._indx == b._indx; };
					friend bool operator!= (const Iterator& a, const Iterator& b) { return a._indx != b._indx; };

					std::vector<act> &get_input_v();
					std::vector<act> &get_output_v();

				private:
					learn_data &_ld;
					uint _indx;
				
			};
			
			Iterator begin() {return Iterator(0,*this);}
			Iterator end() { return Iterator(_ldata.size(),*this); }	// Indice oltre il limite 

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

