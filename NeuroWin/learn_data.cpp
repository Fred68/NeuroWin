
#include "neuro_def.h"
#include "learn_data.h"

namespace neuro
{
	learn_data::learn_data(uint inp_sz, uint out_sz) : _vinp(inp_sz), _vout(out_sz)
	{}
	learn_data::learn_data(const network &net) : learn_data(net.get_input_layer_sz(), net.get_output_layer_sz()) {}


}
