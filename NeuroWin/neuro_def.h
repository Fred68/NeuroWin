#ifndef NEURO_DEF_H
#define NEURO_DEF_H

#define ACT_DBL                     // Definizione del tipo di dato per l'attività: double 
// #undef ACT_DBL                      // Se non definito: float

#define TXT_INFO false              // Informazioni aggiuntive in nodi e sinapsi

#if _DEBUG
#include <iostream>
#define _DEBUG_NEURO_DET false
#endif

#include <string> 

namespace neuro
{

    #ifdef ACT_DBL					// Tipo di dato per l'attività neurale: act
    typedef double act;
    #else
    typedef float act;
    #endif
    
	typedef unsigned int uint;

    enum class FACT { sigmoid = 0, tanh, relu, one, id, Count };

	static std::string fact2string(FACT f)
	{
		std::string str = "";
		switch (f)
		{
		case FACT::one:
			str = "one";
			break;
		case FACT::sigmoid:
			str = "sigmoid";
			break;
		case FACT::tanh:
			str = "tanh";
			break;
		case FACT::relu:
			str = "relu";
			break;
		case FACT::id:
			str = "id";
			break;
		default:
			str = "FACT error";
			break;
		}
		return str;
	}
}

#endif
