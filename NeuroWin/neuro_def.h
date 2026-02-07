
/*************************************************/
/* namespace neuro                               */
/* Implementation for neural network components  */
/* Standard C++ 20.0                             */
/* Version 0.2                                   */
/* Copyright FcSoft november 2025...             */
/* ...gennaio 2026                               */
/* Work in progress...                           */
/*************************************************/



#ifndef NEURO_DEF_H
#define NEURO_DEF_H

#define ACT_DBL                     // Definizione del tipo di dato per l'attività: double 
// #undef ACT_DBL					// Se non definito: float

#define TXT_INFO		false		// Informazioni aggiuntive in nodi e sinapsi

#define TXT_FLOAT_FRM	".4f"

#if _DEBUG	
	#define _DEBUG_NEURO_DET false	// Debug con dettagli
#endif

#define _COPY_CTORS_ false
#define _MOVE_CTORS_ false

#include <iostream>
#include <string> 
#include <format> 
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

	typedef size_t uint;			// In alternativa: typedef unsigned int uint;

    enum class FACT { sigmoid = 0, tanh, relu, one, id, test_error, Count };
	
	enum class EXE_POL { neuron = 0, layer, network };

	static const uint UINT_ERROR = UINT_MAX;		/// Error per uint, equivalente a (uint) -1;

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

	static std::string get_build_time()
	{
		
		return std::format("Compile timestamp: {0}",__TIMESTAMP__);
	}
}

#endif
