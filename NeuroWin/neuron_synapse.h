
#ifndef NEURON_H
#define NEURON_H

#include "neuro_def.h"

#include <vector>
#include <format>
#include <memory>

#include <execution>        // std::execution::par
#include <algorithm>        // for_each
#include <atomic>           // atomic<float>
#include <ranges>			// iota

#define TO_STR_FORMAT_N(n,b) ("x={0:." #n "f},y={1:." #n "f},{5}={4:." #b "f}(f={2}){3}")
#define TO_STR_FORMAT_W(n) ("[{0}{1:." #n "f}]")

namespace neuro
{
	class synapse;

	class neuron
    {

        friend class synapse;
        
        typedef act (*act_func) (neuron*);          // Puntatore a funzione di attivazione

        /// Funzioni di attivazione (non usano dati d'istanza)
        // Scelto argomento neuron*, per usare f(this), invece che neuron& e f(*this) (copia l'oggetto ?)
        static act sigmoid(neuron *n);
        static act sigmoid_der(neuron *n);
        static act hyptangent(neuron *n);
        static act hyptangent_der(neuron *n);
        static act relu(neuron *n);
        static act relu_der(neuron *n);
        static act one(neuron *n);                  // bias modellato come peso di un nodo di uscita unitaria
        static act zero(neuron *n);                 // zero (derivata di costante)
        static act id(neuron *n);                   // identità
        static FACT fact_default() {return FACT::tanh;}
		
		public:
			static constexpr const char *to_string_frm_n = TO_STR_FORMAT_N(3,5);
			static constexpr const char *to_string_frm_w = TO_STR_FORMAT_W(3);

		public:
			static constexpr act w_ini_const = 0.05;
			static constexpr act b_ini_const = 0.001;
			static constexpr act w_ini_mean = 0.5;
			static constexpr act b_ini_mean = 0.001;

        private:
            act x;                                  /// Segnale in ingresso
            act y;                                  /// Attività in uscita
			union
			{										// Union inutile, messa solo per chiarezza
				act beta;							/// beta (primo calcolo), poi...
				act ei;								/// ...EI = beta * F' (secondo calcolo) 			
			};
			#if _DEBUG
			bool isBeta = true;						/// beta or EI
			#endif
            std::vector<synapse> syns;              /// Sinapsi
            FACT fact;                              /// Tipo di funzione di attivazione
            act_func f_act;                         /// Puntatore alla funzione di attivazione
            act_func f_act_der;                     /// Puntatore alla derivata della funzione di attivazione
            bool active = true;                     /// Se false, non calcola né x dai pesi né y.
            bool input = false;                     /// Se true: nodo di input, non calcola la x, solo la y, e abilita set_input
            
			#if TXT_INFO
            std::string name = "";
            #endif

        public:
            neuron();
			neuron(bool isInput);
            neuron(std::vector<neuron> &prev, act std_w = w_ini_const, act bias_w = b_ini_const); 
            ~neuron();

            std::string to_string();

			uint get_n_syn() {return syns.size();}	// Numero di sinapsi
            bool get_active() {return active;}		// Neurone attivo / disattivo		
            void set_active(bool stat);
			
			bool get_input() { return input;}		// Neurone di input o standard
			void set_input(bool inp);				// Non modifica il vettore delle sinapsi
			
			FACT get_fact() {return fact;}			// Funzione di attivazione
			std::string get_fact_name();			// Nome della funzione di attivazione
			void set_fact(FACT f);					// Cambia la funzione di attivazione, solo se non è un nodo di input

			#if TXT_INFO
            std::string get_name() { return name; }
            void set_name(std::string s) { name = s; }
            #endif

			#if _DEBUG
			bool xxx;
			#endif
			
			/// <summary>
			/// Valore dell'ingresso complessivo x
			/// </summary>
			/// <returns></returns>
			act get_x() { return x; }				// Ingresso complessivo				
			/// <summary>
			/// Modifica l'ingresso x, solo se è un nodo di input
			/// </summary>
			/// <param name="x_in"></param>
			/// <returns></returns>
			bool set_x(act x_in);
			/// <summary>
			/// Calcola l'ingresso x, solo se è attivo e non è di input
			/// </summary>
			void calc_x();

			/// <summary>
			/// Valore dell'uscita y
			/// </summary>
			/// <returns></returns>
			act get_y() { return y; }				// Uscita
			/// <summary>
			/// Calcola l'uscita y, solo se è attivo
			/// </summary>
			void calc_y();

			/// <summary>
			/// Valore della derivata dell'errore (ei, in unione con beta)
			/// </summary>
			/// <returns></returns>
			act get_beta();							// Derivata parziale beta dell'errore dE/dy
			/// <summary>
			/// Imposta la derivata dell'errore (beta, in unione con ei)
			/// </summary>
			/// <param name="beta_in"></param>
			void set_beta(act beta_in);
			/// <summary>
			/// Valore della derivata dell'errore (ei, in unione con beta)
			/// </summary>
			/// <returns></returns>
			act get_ei();							// Derivata parziale EI dell'errore dE/dx
			/// <summary>
			/// Imposta la derivata dell'errore (ei, in unione con beta)
			/// </summary>
			/// <param name="ei_in"></param>
			void set_ei(act ei_in);
			/// <summary>
			/// Calcola la derivata EI dell'errore con la formula [7].
			/// Deve essere stata calcolata beta.
			/// </summary>
			void calc_ei();							// Calcola EI con la formula [7]

			/// <summary>
			/// Valore del peso della sinapsi i
			/// </summary>
			/// <param name="i"></param>
			/// <returns></returns>
			act get_w(uint i);						// Peso della sinapsi i.
			/// <summary>
			/// Imposta il peso della sinapsi i
			/// </summary>
			/// <param name="w"></param>
			/// <param name="i"></param>
			void set_w(act w, uint i);
			/// <summary>
			/// Calcolo parziale delle EA = beta dei nodi del livello precedente
			/// Formula [9], ma contributi del nodo j attuale alle beta dei nodi i precedenti
			/// </summary>
			void calc_parz_eai();					// Calcolo parziale delle EA = beta dei nodi del livello precedente
    };
	
	
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
