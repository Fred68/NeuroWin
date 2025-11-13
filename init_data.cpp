

#include "neuro.h"


namespace neuro
{
    /*******************************************/
    /*                                         */
    /* init_data                               */
    /*                                         */
    /*******************************************/

    init_data::init_data(std::vector<int> layers, std::vector<FACT> types)
    {
        this->_layers = layers;
        this->_types = types;
    };

    std::string init_data::to_string()
    {
        std::string str = "";
        char sep = '\0';

        for (int i = 0; i < _layers.size(); i++)
        {
            if (i < _layers.size() - 1)
                sep = '\t';
            else
                sep = '\0';

            str += std::format("{0}{1}", _layers[i], sep);
        }
        str += "\n";
        for (int i = 0; i < _types.size(); i++)
        {
            if (i < _types.size() - 1)
                sep = '\t';
            else
                sep = '\0';

            str += std::format("{0}{1}", network::fact2string(_types[i]), sep);
        }
        return str;
    }
}