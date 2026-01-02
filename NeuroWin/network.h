


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
#include "neuro_exc_static.h"

#include "neuron.h"
#include "init_data.h"
#include "layer.h"
#include "learn_data.h"



#include <string>			// std::format
#include <vector>
#include <memory>
#include <format>
#include <cmath>
#include <tuple>
#include <execution>        // std::execution::par
#include <algorithm>        // for_each
#include <atomic>           // atomic<float>
#include <ranges>			// iota
#include <chrono>			// high_resolution_clock

#define TO_STR_FORMAT_FLOAT(n) ("{0:." #n "f}")

namespace neuro
{
	class learn_data;
	class neuro_exception;

    /*******************************************/
	/*                                         */
    /* network                                 */
	/*                                         */
    /*******************************************/

	// TODO Aggiungere salvataggio e caricamento di una rete (con i pesi), per eseguire l'addestramento in più fasi

    /// <summary>
    /// Class network
    /// </summary>
    class network /*: public std::enable_shared_from_this<network>*/
    {
		public:
			
			/*******************************************/
			// neuro_exception
			/*******************************************/
			class neuro_exception
			{
				friend network;

				public:
					_NEURO_EXC_ENUM;		// Usa la costante con l'enumerazione degli errori
					_NEURO_EXC_STR;			// Usa la costante con le stringhe statiche

				private:
					
					// network *_net è usato solo da network, può essere eliminato.
					// Scartato const network &_net perché mancano network(network &other) e network::operator=(...)
					type _type;
					bool _is_error;
					std::string _desc;
					std::chrono::system_clock::time_point _time;

					/// <summary>
					/// CTOR privato, usato solo da network::create_exception(...)
					/// </summary>
					/// <param name="type"></param>
					/// <param name="is_error"></param>
					/// <param name="desc"></param>
					inline neuro_exception(const type type = type::none, bool is_error = true, std::string desc = "") noexcept :
						_type(type), _is_error(is_error), _desc(desc), _time(std::chrono::system_clock::now()) {}

				public:
					inline neuro_exception(neuro_exception const &other)  noexcept :
									_type(other._type), _is_error(other._is_error), _desc(other._desc), _time(other._time) {}
					neuro_exception& operator=(neuro_exception const &other) noexcept;

					inline static bool is_ex_error(neuro_exception &nex) { return nex._is_error; }
					inline bool is_error() { return _is_error; }
				
				public:
					const std::string what() const noexcept;		// Nessun override di virtual const char* what() const noexcept

			};  // class neuro_exception

		public:
			#ifdef ACT_DBL
				static constexpr act Default_learn_const = 0.01;
			#else
				static constexpr act Default_learn_const = 0.01f;
			#endif
			
			typedef act(*learn_const_func) (network &net, uint iLay, uint iNeu);	// Cost. di apprendim. (puntatore a funzione)
			inline static act lcf_costant_value(network &net, uint iLay, uint iNeu) { return net._learn_const; }
			static std::string display_vector(std::vector<act> &v);

		private:
			typedef void (*lay_func) (std::vector<neuron> &layer, uint i);					// Calcolo di un livello
			typedef act (*weight_func) (uint iLay, uint iNeu, uint iSyn, bool is_bias);		// Inizializzazione di un peso
		
			//const std::shared_ptr<network> _this;	// shared this 
			std::vector<layer> _layers;				// Modificato solo nel ctor
            uint _nLays = 0;
			uint _nInputs = 0;
			uint _nOutputs = 0;
			
			act _learn_const = Default_learn_const;
			learn_const_func _learn_const_pf;				// Puntatore alla funzione che restituisce la costante di apprendimento	

			// std::execution::seq = sequenziale, singolo thread. Nessun 'data race'
			// std::execution::par = parallelo su più thread. Evitare 'data race' con mutex o atomic. 
			// std::execution::unseq = vettorizzato su singolo thread. Usa istruzioni che lavorano su più dati insieme.
			// std::execution::par_unseq = vettorizzato su più thread. 
			// Con unseq lo stesso thread potrebbe scrivere simultaneamente (con unica istruzione che agisce su più dati).
			// In ogni caso un mutex introduce un overhead. Se l'operazione è semplice, meglio usare dati atomici
			std::execution::parallel_policy exe_pol[3] = {std::execution::par, std::execution::par, std::execution::par};

			std::vector<neuro_exception> _exceptions;

            /// <summary>
            /// Neurone del livello 'lay' e con indice 'num'.
			/// Indici non controllati
            /// </summary>
            /// <param name="lay"></param>
            /// <param name="num"></param>
            /// <returns></returns>
			inline neuron &get_at(uint lay, uint num) {return (_layers[lay])[num];}	// No check indici
            


			#if TXT_INFO
            void name_elements();
            #endif

			/// <summary>
			/// Imposta gli ingressi. Lunghezza del vettore non controllata.
			/// </summary>
			/// <param name="inp_lay"></param>
			/// <returns></returns>
			bool set_inputs(const std::vector<act> &inp_lay);
			/// <summary>
			/// Imposta le uscite. Lunghezza del vettore non controllata.
			/// Aggiorna _err_tot
			/// </summary>
			/// <param name="out_lay"></param>
			/// <returns></returns>
			bool set_outputs(const std::vector<act> &out_lay, act &error_tot);
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
			/// <summary>
			/// Ricalcola i pesi delle sinapsi dei nodi del livello nlay
			/// </summary>
			/// <param name="nlay"></param>
			/// <returns></returns>
			void calc_w_lay(uint nlay);					// Ricalcola i pesi delle sinapsi dei nodi del livello nlay (se recalc_w è vero)

			/// <summary>
			/// Calcola la rete con forward propagation, partendo dai valori del vettore di input.
			/// Per ogni nodo (dal primo all'ultimo livello)...
			/// ...calcola lingresso totale (x) e attività di uscita (y), azzera EI.</summary>
			/// <param name="inp_lay"></param>
			/// <returns></returns>
			bool prop_fw(const std::vector<act> &inp_lay);		// Calcola singola forward propagation	
			/// <summary>
			/// Calcola singola back propagation.
			/// Per ogni nodo(dall'ultimo al primo livello)...
			/// ...calcola le EI.
			/// </summary>
			/// <param name="out_lay"></param>
			/// <param name="error_tot"></param>
			/// <returns></returns>
			bool prop_bw(const std::vector<act> &out_lay, act &error_tot);		// Calcola singola back propagation
			/// <summary>
			/// Corregge i pesi
			/// </summary>
			void update_w();								// Aggiorna i pesi
			/// <summary>
			/// Calcola back-propagation senza controllare il numero di nodi dei livelli
			/// </summary>
			/// <param name="inp_lay"></param>
			/// <param name="out_lay"></param>
			/// <param name="cycles"></param>
			/// <param name="error_tot">errore totale (al quadrato)</param>
			/// <returns></returns>
			bool backward_propagate_no_check(const std::vector<act> &inp_lay, const std::vector<act> &out_lay, uint cycles, act &error_tot);

        public:
            /// <summary>
            /// Ctor con dati di inizializzazione
            /// </summary>
            /// <param name="ini_data"></param>
            network(init_data &ini_data);
            /// <summary>
            /// dtor
            /// </summary>
            ~network();
            /// <summary>
            /// to_string
            /// </summary>
            /// <returns></returns>
            std::string to_string();
			/// <summary>
			/// Reference a *this
			/// </summary>
			/// <returns></returns>
			inline network &get_reference() {return *this;}
			/// <summary>
			/// Crea un'eccezione (e la aggiunge all'elenco), non esegue alcun throw
			/// </summary>
			/// <param name="type"></param>
			/// <param name="is_error"></param>
			/// <param name="desc"></param>
			/// <returns></returns>
			constexpr neuro_exception &create_exception(const neuro_exception::type type = neuro_exception::type::none, bool is_error = true, std::string desc = "");
			/// <summary>
			/// Svuota l'elenco delle eccezioni
			/// </summary>
			void clear_exceptions();
			/// <summary>
			/// Controlla se, nella lista delle eccezioni, ci sono errori (o soltanto avvertimenti)
			/// </summary>
			/// <returns></returns>
			bool isOk();
			/// <summary>
			/// Stringa con l'elenco delle eccezioni
			/// </summary>
			/// <param name="show_warnings">se true, include gli avvertimenti</param>
			/// <returns></returns>
			std::string get_exceptions_string(bool show_warnings = false);
				

			inline uint get_n_layers() const {return _nLays;}
			inline uint get_input_layer_size() const { return _nInputs; }
			inline uint get_output_layer_size() const { return _nOutputs; }

			inline act get_learn_const() const {return _learn_const;}
			inline void set_learn_const(act lrn_c) {_learn_const = lrn_c;}
			inline std::execution::parallel_policy get_exe_pol(EXE_POL pol) const { return exe_pol[(int)pol]; }
			
			inline learn_const_func get_f_learn() const { return _learn_const_pf; };
			inline void set_f_learn(learn_const_func lrn_f) { _learn_const_pf = lrn_f; };
			
            /// <summary>
            /// Riferimento al neurone del livello 'lay' e con indice 'num'
			/// Se indici errati: eccezione.
            /// </summary>
            /// <param name="lay"></param>
            /// <param name="num"></param>
            /// <returns></returns>
            neuron &get_neuron(uint lay, uint num);			// Riferimento al neurone del livello 'lay' e con indice 'num'

			/// <summary>
			/// Esegue una forward propagation per calcolare i valori di uscita
			/// </summary>
			/// <param name="inp_lay">valori di ingresso</param>
			/// <param name="out_lay">valori</param>
			/// <returns></returns>
			bool forward_propagate(const std::vector<act> &inp_lay, std::vector<act> &out_lay);
			/// <summary>
			/// Esegue cicli di ricalcolo back propagation, aggiornando i pesi
			/// per approssimare i valori in uscita partendo dai valori di ingresso
			/// </summary>
			/// <param name="inp_lay">valori di ingresso</param>
			/// <param name="out_lay">valori di uscita desiderati</param>
			/// <param name="cycles">numero di cicli</param>
			/// <param name="error_tot">errore totale (al quadrato)</param>
			/// <param name="msec_elap">millisecondi impiegati</param>
			/// <returns></returns>
			bool backward_propagate(const std::vector<act> &inp_lay, const std::vector<act> &out_lay, uint cycles, act &error_tot, std::chrono::milliseconds &msec_elap);
			/// <summary>
			/// Ripete cicli ricalcolo dei pesi (con back propagation) sui dati di apprendimento
			/// </summary>
			/// <param name="pldata">dati di apprendimento</param>
			/// <param name="cycles">numero di cicli di apprendimento</param>
			/// <param name="subcycles">numero di ripetizioni per ogni caso</param>
			/// <param name="error_med">errore quadratico medio</param>
			/// <param name="msec_elap">millisecondi impiegati</param>
			/// <returns></returns>
			bool backward_propagate(std::shared_ptr<learn_data> pldata, const uint cycles, const uint subcycles, act &error_med, std::chrono::milliseconds &msec_elap);

    };  // class network



}  // namespace neuro

#endif // NETWORK_H
