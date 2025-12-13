
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
		
		std::vector<int> _layers;			/// Vettore con il numero di nodi per livello
		std::vector<FACT> _types;			/// Vettore con i tipi di funzioni di attivazione per livello
		act _learn_const;

		bool check();

	public:
		
		init_data(std::vector<int> layers, std::vector<FACT> types, act learn_const_data);
		
		std::string to_string();

		std::vector<int> const &get_layers() {return _layers;}
		std::vector<FACT> const &get_types() { return _types;}
		act get_learn_const() {return _learn_const;}
		void set_learn_const(act lc) {_learn_const = lc;}
		bool is_ok() {return _ok;}
		uint get_layers_num() {return (_ok ? _layers.size() : 0);}
	};

}
