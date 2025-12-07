
#include "neuro_def.h"

#include <vector>
#include <format>


namespace neuro
{
	class init_data
	{
	public:
		std::vector<int> _layers;
		std::vector<FACT> _types;
		init_data(std::vector<int> layers, std::vector<FACT> types, act learn_const_data);
		std::string to_string();
		act _learn_const;
	};

}
