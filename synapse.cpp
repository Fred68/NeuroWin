
#include "neuro.h"


namespace neuro
{

    /*******************************************/
    /*                                         */
    /* synapse                                 */
    /*                                         */
    /*******************************************/

    synapse::synapse()
    {
        pn = std::shared_ptr<neuron>(nullptr);  // Non usa pn=std::make_shared<neuron>() perché alloca un nuovo oggetto
        w = (act) 1;
        #if _DEBUG_NEURO_DET
        cout << "synapse()\n";
        #endif
    }
    synapse::synapse(neuron &p_n, act ws)
    {
        pn = std::shared_ptr<neuron>(&p_n);     // Non usa pn=std::make_shared<neuron>(p_n) perché crea una copia
        w = ws;
        #if _DEBUG_NEURO_DET
        cout << "synapse(p_n)\n";
        #endif
    }
    synapse::~synapse()
    {
        #if _DEBUG_NEURO_DET
        cout << "~synapse()\n";
        getchar();
        #endif
    }


}