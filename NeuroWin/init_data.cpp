
#include "init_data.h"


namespace neuro
{
    /*******************************************/
    /*                                         */
    /* init_data                               */
    /*                                         */
    /*******************************************/

    init_data::init_data(std::vector<int> layers, std::vector<FACT> types, act learn_const_data) : _layers(layers), _types(types), _learn_const(learn_const_data)
    {	
		_ok = check();
    }		

	bool init_data::check()
	{
		bool ok = true;

		#ifdef ACT_DBL
		if(std::abs(_learn_const) < EPSILON)
		#else
		if (std::fabs(_learn_const) < EPSILON)
		#endif
		{
			err += "Learn const is null";
			ok = false;
		}
		if (_layers.size() != _types.size())
		{
			err += "Layer and type vectors sizes don't match";
			ok = false;
		}

		if (_layers.size() < 2)
		{
			err += "Minimum 2 layers required";
			ok = false;
		}

		return ok;
	}

    std::string init_data::to_string()
    {
        std::string str = "";
        char sep = '\0';
		
		if(!_ok)
		{
			str += std::format("init_data is not valid");
		}
		
		str += std::format("learning const = {0}",_learn_const);

		str += "\n";
		for (int i = 0; i < _layers.size(); i++)
        {
            if (i < _layers.size() - 1)
                sep = '\n';
            else
                sep = '\0';

            str += std::format("layer[{3}]: N.{0} {1}{2}", _layers[i], fact2string(_types[i]), sep,i);
        }
        
        return str;
    }
}