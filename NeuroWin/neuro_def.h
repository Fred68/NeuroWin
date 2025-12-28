

#ifndef NEURO_DEF_H
#define NEURO_DEF_H

#define ACT_DBL                     // Definizione del tipo di dato per l'attività: double 
// #undef ACT_DBL					// Se non definito: float

#define TXT_INFO		false		// Informazioni aggiuntive in nodi e sinapsi

#define TXT_FLOAT_FRM	".4f"
#if _DEBUG	
	#define _DEBUG_NEURO_DET false
#endif

//#include "network.h"

#include <iostream>
#include <string> 
#include <execution>
#include <limits>

namespace neuro
{

    #ifdef ACT_DBL					// Tipo di dato per l'attività neurale: act
		typedef double act;
    #else
		typedef float act;
    #endif
	
	#define EPSILON 1E-12

	typedef size_t uint;		// In alternativa: typedef unsigned int uint;

    enum class FACT { sigmoid = 0, tanh, relu, one, id, Count };
	
	enum class EXE_POL { neuron = 0, layer, network };

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

#include "PIPPO.h"
//#include "neuro_exception.h"

#endif
