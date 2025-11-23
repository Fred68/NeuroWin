
#include "neuro_def.h"

#include <vector>
#include <format>


#if true
namespace neuro
{
	class init_data
	{
	public:
		std::vector<int> _layers;
		std::vector<FACT> _types;
		init_data(std::vector<int> layers, std::vector<FACT> types);
		std::string to_string();

	};

}
#endif