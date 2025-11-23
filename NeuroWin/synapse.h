
#ifndef SYNAPSE_H
#define SYNAPSE_H

#if false

#include "neuro_def.h"
#include "neuron.h"

#include <memory>

namespace neuro
{
	class neuron;

	class synapse
	{
		friend class neuron;

		private:
			std::shared_ptr<neuron> pn;
			act    w;

		public:
			synapse();
			synapse(neuron &p_n, act ws);
			~synapse();
			act x() { return w * pn.get()->y; }
	};

}
#endif
#endif

