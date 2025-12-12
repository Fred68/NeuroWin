
#include "neuro_def.h"

#include <vector>
#include <format>


namespace neuro
{
	class init_data
	{
	public:
		std::vector<int> _layers;			/// Vettore con il numero di nodi per livello
		std::vector<FACT> _types;			/// Vettore con i tipi di funzioni di attivazione per livello
		init_data(std::vector<int> layers, std::vector<FACT> types, act learn_const_data);
		std::string to_string();
		act _learn_const;					// Costante di apprendimento
	};

}
