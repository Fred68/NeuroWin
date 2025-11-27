
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


namespace neuro
{
	class synapse;

	class neuron
    {

        friend class synapse;
        
        typedef act (*act_func) (neuron*);          // Puntatore a funzione di attivazione

        // Funzioni di attivazione (non usano dati d'istanza)
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
			static constexpr act w_ini_const = 0.05;
			static constexpr act b_ini_const = 0.001;
			static constexpr act w_ini_mean = 0.5;
			static constexpr act b_ini_mean = 0.001;



        private:
            act x;                                  // Segnale in ingresso
            act y;                                  // Attività in uscita
			act b_ei;								// Valori beta oppure EI (in base al ciclo dell'algoritmo)
            std::vector<synapse> syns;              // Sinapsi
            FACT fact;                              // Tipo di funzione di attivazione
            act_func f_act;                         // Puntatori alla funzione di attivazione e...
            act_func f_act_der;                     // ...alla sua derivata.
            bool active = true;                     // Se false, non calcola né x dai pesi né y.
            bool input = false;                     // Se true: nodo di input, non calcola la x, solo la y, e abilita set_input
            
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

			act get_x() { return x; }				// Ingresso complessivo
			bool set_x(act x_in);                   // Modifica l'ingresso x, solo se è un nodo di input. Se no restituisce false.
			void calc_x();                          // Calcola x, solo se è active e se non è un nodo di input

			act get_y() { return y; }				// Uscita
			void calc_y();                          // Calcola y, solo se active

			act get_ei() { return b_ei; }			// EI, derivata dell'errore
			void set_ei(act b_in);

			act get_w(uint i);						// Sinapsi i.
			void set_w(act w, uint i);
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
