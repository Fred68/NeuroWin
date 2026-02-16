
#include "neuro_def.h"

#include <vector>
#include <format>


namespace neuro
{
	class init_data
	{
	private:
		
		bool _ok = false;
		std::string err = "";
		
		std::vector<uint> _layers;			/// Vettore con il numero di nodi per livello
		std::vector<FACT> _types;			/// Vettore con i tipi di funzioni di attivazione per livello
		act _learn_const;

		bool check();

	public:
		
		init_data(std::vector<uint> layers, std::vector<FACT> types, act learn_const_data);
		
		std::string to_string();

		std::vector<uint> const &get_layers() {return _layers;}
		std::vector<FACT> const &get_types() { return _types;}
		inline act get_learn_const() {return _learn_const;}
		inline void set_learn_const(act lc) {_learn_const = lc;}
		inline bool is_ok() {return _ok;}
		inline uint get_layers_num() {return (_ok ? _layers.size() : 0);}
		uint get_input_size();
		uint get_output_size();

	};

}
