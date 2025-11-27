


/*************************************************/
/* neuro.cpp                                     */
/* Implementation for neural network components  */
/* Standard C++ 20.0                             */
/* Version 0.1                                   */
/* Copyright FcSoft november 2025                */
/* Work in progress...                           */
/*************************************************/



#ifndef NETWORK_H
#define NETWORK_H

#include "neuro_def.h"
#include "neuron.h"

#include <string>
#include <vector>
#include <memory>
#include <format>
#include <cmath>
#include <tuple>
#include <execution>        // std::execution::par
#include <algorithm>        // for_each
#include <atomic>           // atomic<float>
#include <ranges>			// iota

#include "init_data.h"


namespace neuro
{
  
    /*******************************************/
    // network
    /*******************************************/

    /// <summary>
    /// Class network
    /// </summary>
    class network
    {
		private:
			// Puntatori a funzione
			typedef void (*lay_func) (std::vector<neuron> &layer, uint i);					// Calcolo di un livello
			typedef act (*weight_func) (uint iLay, uint iNeu, uint iSyn, bool is_bias);		// Inizializzazione di un peso

        private:
            uint _nLays = 0;
            std::vector<std::vector<neuron>> _layers;
			std::vector<act> _beta_out;

        private:
            neuron &get_at(uint lay, uint num) {return (_layers[lay])[num];}	// No check indici
            #if TXT_INFO
            void name_elements();
            #endif
			bool set_inputs(std::vector<act> &inp_lay);			// Imposta gli ingressi (check indici)
			bool set_outputs(std::vector<act> &out_lay);
			void set_weights(weight_func wf);					// Imposta i pesi iniziali
			// Funzioni per i pesi iniziali
			act set_w_const(uint iLay, uint iNeu, uint iSyn, bool is_bias);		// Pesi e bias costanti
			act set_w_mean(uint iLay, uint iNeu, uint iSyn, bool is_bias);		// Pesi e bias medi (no check indici)


			bool calc_y_lay(uint nlay);							// Calcola le attività e azzera EI (no check indici)

			bool calc_b_lay(uint nlay);							// Calcola le derivate dell'errore (no check indici) [DA COMPLETARE]


        public:
            network(init_data &ini_data);
            ~network();
            std::string to_string();
            neuron &get_neuron(uint lay, uint num);
            
			bool prop_fw(std::vector<act> &inp_lay);			// Calcola forward propagation
			bool prop_bw(std::vector<act> &out_lay);			// Calcola back propagation [DA SCRIVERE]

    };  // class network

}  // namespace neuro

#endif // NETWORK_H
