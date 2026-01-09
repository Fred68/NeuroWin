


#include "neuron.h"



namespace neuro
{
    /*******************************************/
    /*                                         */
    /* synapse                                 */
    /*                                         */
    /*******************************************/

    neuron::synapse::synapse()
    {
        _pn = std::shared_ptr<neuron>(nullptr);  // Non usa _pn=std::make_shared<neuron>() perché allocherebbe un nuovo oggetto
		_in = UINT_ERROR;
        w = (act) 1;
        #if _DEBUG_NEURO_DET
        cout << "synapse()\n";
        #endif
    }
	neuron::synapse::synapse(neuron &p_n, act ws)
    {
		_pn = std::shared_ptr<neuron>(&p_n);     // Non usa _pn=std::make_shared<neuron>(p_n) perché creerebbe una copia
		_in = UINT_ERROR;
        w = ws;
        #if _DEBUG_NEURO_DET
        cout << "synapse(p_n)\n";
        #endif
    }
	neuron::synapse::~synapse()
    {
        #if _DEBUG_NEURO_DET
        cout << "~synapse()\n";
        getchar();
        #endif
    }

	void neuron::synapse::write(std::ofstream &fs)
	{
		try
		{
			uint neuron_indx = _pn->get_index();		// Era uint neuron_indx = std::get<ptN>(_pn)->get_index();
			fs.write(reinterpret_cast<char*>(&neuron_indx), sizeof(neuron_indx));
			fs.write(reinterpret_cast<char*>(&w), sizeof(w));
		}
		catch (std::exception &ex)
		{
			std::cerr << "Errore in synapse::write(...): " << ex.what() << std::endl;
		}
	}

	void neuron::synapse::read(std::ifstream &fs)
	{
		try
		{
			uint i_tmp;
			act w_tmp;
			fs.read(reinterpret_cast<char*>(&i_tmp), sizeof(i_tmp));
			fs.read(reinterpret_cast<char*>(&w_tmp), sizeof(w_tmp));
			_in = i_tmp;
			_pn = ptN(nullptr);
			w = w_tmp;
		}
		catch (std::exception &ex)
		{
			std::cerr << "Errore in synapse::read(...): " << ex.what() << std::endl;
		}

	}

}