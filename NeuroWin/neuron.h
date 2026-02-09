
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
#include <variant>			// union non anonima in classe usata in un vettore sarebbe più complessa.

#include <fstream>			// I/O su stream (binario)


// Formato per neuron::to_string(), con stat::ei o stat::beta
#define TO_STR_FORMAT_N(n,b) ("x={0:." #n "f},y={1:." #n "f},{5}={4:." #b "f}(f={2}){3}")
#define TO_STR_FORMAT_N_INDX(n) ("x={0:." #n "f},y={1:." #n "f},{5}={4}(f={2}){3}")
#define TO_STR_FORMAT_W(n) ("[{0}{1:." #n "f}]")

namespace neuro
{

	class neuron;
	class network;
	class synapse;	
	class pippo;

	typedef std::shared_ptr<neuron> ptN;

	class neuron
    {
		enum stat : char { _beta, _ei, _index };

		class synapse
		{
			friend class neuron;

			private:
				uint _in;			// Indice del neurone collegato
				ptN _pn;			// Puntatore al neurone collegato (era std::variant<ptN,uint> _pn)
				act w;
		
				void write(std::ofstream &fs);
				void read(std::ifstream &fs);

			public:
				/// <summary>
				/// default ctor
				/// </summary>
				synapse();							// default ctor
				/// <summary>
				/// ctor
				/// </summary>
				/// <param name="p_n">Reference al neurone, viene trasformato in shared_ptr<neuron></param>
				/// <param name="ws">peso</param>
				synapse(neuron &p_n, act ws);		// ctor
				
				#if _DEBUG_DTOR
				~synapse() noexcept;				// dtor
				#endif

				//inline void clear_ptrs()
				//{
				//	_pn.reset();
				//}

				void reset();
				#if _COPY_CTORS_
				synapse(const synapse& other);
				synapse& operator=(const synapse& other);
				#endif
				#if _MOVE_CTORS_
				synapse(synapse&& other);
				synapse& operator=(synapse&& other);
				// Necessari costruttore di copia e assegnazione di copia standard (copia bit per bit).
				// synapse(const synapse&)=delete e synapse& operator=(const synapse&)=delete generano errore di compilazione.
				#endif

				/// <summary>
				/// Aggiorna l'indice del neurone a cui fa riferimento la sinapsi
				/// </summary>
				/// <returns>false se errore</returns>
				bool update_node_index();

				/// <summary>
				/// Imposta l'indice del neurone a cui fa riferimento la sinapsi, senza alcun controllo.
				/// Azzera il puntatore al neurone
				/// </summary>
				/// <param name="i"></param>
				void set_node_index(uint i);

				/// <summary>
				/// Imposta il puntatore al neurone a cui fa riferimento la sinapsi, senza alcun controllo.
				/// Non modifica l'indice.
				/// </summary>
				
				inline void set_node_ptr(ptN ptn) { _pn = ptn;}


				/// <summary>
				/// Restituisce l'indice del neurone della sinapsi
				/// UINT_ERROR se non impostato
				/// </summary>
				/// <returns></returns>
				inline uint get_node_index() {return _in;}
		};

        typedef act (*act_func) (neuron*);							// Puntatore a funzione di attivazione

        /// Funzioni di attivazione (non usano dati d'istanza)
        // Scelto argomento neuron*, per usare f(this), invece che neuron& e f(*this) (copia l'oggetto ?)
        static act sigmoid(neuron *n);
        static act sigmoid_der(neuron *n);
        static act hyptangent(neuron *n);
        static act hyptangent_der(neuron *n);
        static act relu(neuron *n);
        static act relu_der(neuron *n);
        static act one(neuron *n);                  // bias modellato come peso di un neurone di uscita unitaria
        static act zero(neuron *n);                 // zero (derivata di costante)
        static act id(neuron *n);                   // identità
        inline static FACT fact_default() {return FACT::tanh;}

		public:
			static constexpr const char *to_string_frm_n = TO_STR_FORMAT_N(3,5);
			static constexpr const char *to_string_frm_n_indx = TO_STR_FORMAT_N_INDX(3);
			static constexpr const char *to_string_frm_w = TO_STR_FORMAT_W(3);


		public:
			static constexpr act w_ini_const = 0.05;
			static constexpr act b_ini_const = 0.001;
			static constexpr act w_ini_mean = 0.5;
			static constexpr act b_ini_mean = 0.001;

        private:
			network &_net;							/// Riferimento alla rete di appartenenza
            act x;                                  /// Segnale in ingresso
            act y;                                  /// Attività in uscita
			union
			{
				uint index_in_layer;				/// Per I/O
				act beta;							/// beta (primo calcolo), poi...
				act ei;								/// ...EI = beta * F' (secondo calcolo)
			};
            std::vector<synapse> _syns;              /// Sinapsi
            act_func f_act;                         /// Funzione di attivazione (puntatore)
            act_func f_act_der;                     /// Derivata della funzione di attivazione (puntatore)
			
			FACT _fact;                              /// Tipo di funzione di attivazione
            bool _active = true;                     /// Se false, non calcola né x dai pesi né y.
            bool _input = false;                     /// Se true: neurone di _input, non calcola la x, solo la y, e abilita set_input
			stat _nstat = stat::_beta;				/// beta, EI o index (for I/O)

			#if TXT_INFO
            std::string name = "";
            #endif

        public:
			/// <summary>
			/// ctor, neurone vuoto
			/// </summary>
			/// <param name="netwrk">rete (riferimento)</param>
			neuron(network &netwrk);
			/// <summary>
			/// ctor, neurone vuoto
			/// </summary>
			/// <param name="netwrk">rete (riferimento)</param>
			/// <param name="isInput">neurone di _input se true</param>
			neuron(network &netwrk, bool isInput);
			/// <summary>
			/// ctor, crea sinapsi a tutti i neuroni del livello precedente
			/// </summary>
			/// <param name="netwrk">rete (riferimento)</param>
			/// <param name="prev">neuroni del livello precedente</param>
			/// <param name="neu_w">peso</param>
			/// <param name="bias_w">bias</param>
			neuron(network &netwrk, std::vector<neuron> &prev, act neu_w = w_ini_const, act bias_w = b_ini_const);
			/// <summary>
			/// ctor, crea sinapsi ai soli neuroni del livello precedente con gli indici richiesti
			/// </summary>
			/// <param name="netwrk">rete (riferimento)</param>
			/// <param name="prev">neuroni del livello precedente</param>
			/// <param name="indx">indici dei neuroni del livello precedente da collegare con sinapsi</param>
			/// <param name="neu_w">peso</param>
			/// <param name="bias_w">bias</param>
			neuron(network &netwrk, std::vector<neuron> &prev, std::vector<uint> &indx, act neu_w = w_ini_const, act bias_w = b_ini_const);
			
			#if _COPY_CTORS_
			neuron(const neuron& other);
			neuron& operator=(const neuron& other);
			#endif
			#if _MOVE_CTORS_
			neuron(neuron&& other);
			neuron& operator=(neuron&& other);
			#endif

			#if _DEBUG_DTOR
			~neuron() noexcept;
			#endif

			//inline void clear_ptrs()
			//{
			//	for(uint i=0; i<_syns.size(); i++)
			//	{
			//		_syns[i].clear_ptrs();
			//	}
			//}

			void reset();

            std::string to_string();

			inline uint get_n_syn() const {return _syns.size();}	// Numero di sinapsi
			inline bool get_active() const {return _active;}		// Neurone attivo / disattivo		
            void set_active(bool stat);
			
			inline bool get_input() const { return _input;}		// Neurone di _input o standard
			void set_input(bool inp);							// Non modifica il vettore delle sinapsi
			
			inline FACT get_fact() {return _fact;}				// Funzione di attivazione
			std::string get_fact_name();						// Nome della funzione di attivazione
			void set_fact(FACT f);								// Cambia la funzione di attivazione, solo se non è un neurone di input
			
			#if TXT_INFO
			inline std::string get_name() { return name; }
			inline void set_name(std::string s) { name = s; }
            #endif

			/// <summary>
			/// Aggiunge una sinapsi con indice l'indice del nodo a cui fa riferimento,
			/// senza alcun controllo.
			/// </summary>
			/// <param name="in"></param>
			// TODO: funzione da controllare
			void add_synapse(uint in);

			/// <summary>
			/// Valore dell'ingresso complessivo x
			/// </summary>
			/// <returns></returns>
			/// 
			inline act get_x() { return x; }

			/// <summary>
			/// Modifica l'ingresso x, solo se è un neurone di input
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
			inline act get_y() { return y; }		// Uscita

			/// <summary>
			/// Calcola l'uscita y, solo se è attivo
			/// </summary>
			void calc_y();

			/// <summary>
			/// Indice del neurone nel livello (per salvataggio)
			/// </summary>
			/// <returns></returns>
			uint get_index();
			
			/// <summary>
			/// Imposta l'indice del neurone nel livello e delle sue sinapsi (per salvataggio)
			/// </summary>
			/// <param name="i"></param>
			/// <returns></returns>
			void set_index(uint indx);
			
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
			/// Valore del peso della sinapsi 'i'
			/// </summary>
			/// <param name="i"></param>
			/// <returns></returns>
			act get_w(uint i);						// Peso della sinapsi i.
			
			/// <summary>
			/// Indice del neurone della sinapsi 'i'
			/// </summary>
			/// <param name="i"></param>
			/// <returns></returns>
			uint get_neuron_index(uint i);			// Indice del neurone della sinapsi i.

			/// <summary>
			/// Imposta il peso della sinapsi i
			/// </summary>
			/// <param name="w"></param>
			/// <param name="i"></param>
			void set_w(act w, uint i);
			
			/// <summary>
			/// Calcolo parziale delle EA = beta dei nodi del livello precedente
			/// Formula [9], ma contributi del neurone j attuale alle beta dei nodi i precedenti
			/// </summary>
			void calc_parz_eai();					// Calcolo parziale delle EA = beta dei nodi del livello precedente
			
			/// <summary>
			/// Ricalcola i pesi (riceve da network la costante di apprendimento)
			/// </summary>
			/// <param name="learn_const"></param>
			void calc_w(act learn_const);			// Ricalcolo dei pesi (riceve da network la costante di apprendimento)

			/// <summary>
			/// Aggiorna, nelle sinapsi, i puntatori ai neuroni, in base agli indici
			/// </summary>
			/// <param name="ilay"></param>
			void update_syn_pointers(uint ilay);	// Aggiorna, nelle sinapsi, i puntatori ai neuroni

			void write(std::ofstream &fs);
			void read(std::ifstream &fs);
    };
	
	

}

#endif
