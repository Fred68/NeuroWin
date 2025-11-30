


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
#include "neuron_synapse.h"

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
            /// <summary>
            /// Neurone del livello 'lay' e con indice 'num'.
			/// Indici non controllati
            /// </summary>
            /// <param name="lay"></param>
            /// <param name="num"></param>
            /// <returns></returns>
            neuron &get_at(uint lay, uint num) {return (_layers[lay])[num];}	// No check indici
            #if TXT_INFO
            void name_elements();
            #endif
			/// <summary>
			/// Imposta gli ingressi. Lunghezza del vettore non controllata.
			/// </summary>
			/// <param name="inp_lay"></param>
			/// <returns></returns>
			bool set_inputs(std::vector<act> &inp_lay);
			/// <summary>
			/// Imposta le uscite. Lunghezza del vettore non controllata.
			/// </summary>
			/// <param name="out_lay"></param>
			/// <returns></returns>
			bool set_outputs(std::vector<act> &out_lay);
			/// <summary>
			/// Calcola i pesi iniziali usando il puntatore a funzione 'wf'
			/// </summary>
			/// <param name="wf"></param>
			void set_weights(weight_func wf);
			/// <summary>
			/// Funzione per impostare i pesi iniziali (valori costanti)
			/// /// Indici non controllati
			/// </summary>
			/// <param name="iLay">indice del livello</param>
			/// <param name="iNeu">indice del neurone</param>
			/// <param name="iSyn">indice della sinapsi</param>
			/// <param name="is_bias">E' un peso o un bias</param>
			/// <returns></returns>
			act set_w_const(uint iLay, uint iNeu, uint iSyn, bool is_bias);		// Pesi e bias costanti
			/// <summary>
			/// Funzione per impostare i pesi iniziali (valori medi)
			/// Indici non controllati
			/// </summary>
			/// <param name="iLay">indice del livello</param>
			/// <param name="iNeu">indice del neurone</param>
			/// <param name="iSyn">indice della sinapsi</param>
			/// <param name="is_bias">E' un peso o un bias</param>
			/// <returns></returns>
			act set_w_mean(uint iLay, uint iNeu, uint iSyn, bool is_bias);		// Pesi e bias medi (no check indici)

			/// <summary>
			/// Calcola la attività y di un livello ed azzera i valori di beta
			/// Non controlla gli indici
			/// </summary>
			/// <param name="nlay"></param>
			/// <returns></returns>
			bool calc_y_lay(uint nlay);					// Calc. le attività y del livello nlay e azzera le beta
			/// <summary>
			/// Calcola le derivate EI dell'errore del livello nLay e le beta del livello precedente
			/// Non controlla gli indici
			/// </summary>
			/// <param name="nlay"></param>
			/// <returns></returns>
			bool calc_ei_eaprec_lay(uint nlay);			// Calcola le derivate EI dell'errore del livello nLay e le beta del prec.


        public:
            network(init_data &ini_data);
            ~network();
            std::string to_string();
            /// <summary>
            /// Riferimento al neurone del livello 'lay' e con indice 'num'
			/// Se indici errati: eccezione.
            /// </summary>
            /// <param name="lay"></param>
            /// <param name="num"></param>
            /// <returns></returns>
            neuron &get_neuron(uint lay, uint num);
			/// <summary>
			/// Calcola la rete con forward propagation, partendo dai valori del vettore di input
			/// Per ogni nodo (dal primo all'ultimo livello)...
			/// calcola lingresso totale (x) e attività di uscita (y), azzera EI.
			/// </summary>
			/// <param name="inp_lay"></param>
			/// <returns></returns>
			bool prop_fw(std::vector<act> &inp_lay);			// Calcola forward propagation
			bool prop_bw(std::vector<act> &out_lay);			// Calcola back propagation [DA SCRIVERE]

    };  // class network

}  // namespace neuro

#endif // NETWORK_H
