

#include "test.h"

namespace neuro
{
	std::string test::to_string()
	{
		return std::format("x={}, y={}",_x,_y);
	}

}