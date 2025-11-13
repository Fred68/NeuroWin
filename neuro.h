

/*************************************************/
/* neuro.cpp                                     */
/* Implementation for neural network components  */
/* Standard C++ 20.0                             */
/* Version 0.1                                   */
/* Copyright FcSoft                              */
/*************************************************/



#ifndef NEURO_H                     //#pragma once
#define NEURO_H

#define ACT_DBL                     // Definizione del tipo di dato per l'attività: double 
// #undef ACT_DBL                      // Se non definito: float

#define TXT_INFO false              // Informazioni aggiuntive in nodi e sinapsi

#if _DEBUG
    #include <iostream>
    #define _DEBUG_NEURO_DET false
#endif

#include <string>
#include <vector>
#include <memory>

#include <format>
#include <cmath>
#include <tuple>
#include <execution>        // std::execution::par
#include <algorithm>        // for_each
#include <atomic>           // atomic<float>

namespace neuro
{

    // Tipo di dato per l'attività neurale: act
    #ifdef ACT_DBL
    typedef double act;
    #else
    typedef float act;
    #endif
    
    enum class FACT { sigmoid = 0, tanh, relu, one, id, Count };

    // Forward declarations
    class neuron;
    class synapse;
    class init_data;
    class network;
    class test;


    /*******************************************/
    // init_data
    /*******************************************/

    class init_data
    {
    public:
        std::vector<int> _layers;
        std::vector<FACT> _types;
        init_data(std::vector<int> layers, std::vector<FACT> types);
        std::string to_string();

    };


    
    
    /*******************************************/
    // network
    /*******************************************/

    /// <summary>
    /// Class network
    /// </summary>
    class network
    {

        private:
            unsigned int _nLays = 0;
            std::vector<std::vector<neuron>> _layers;

        private:
            neuron& get_at(unsigned int lay, unsigned int num) {return (_layers[lay])[num];}
            #if TXT_INFO
            void name_elements();
            #endif


        public:
            network();
            network(init_data &ini_data);
            ~network();
            std::string to_string();
            neuron& get_neuron(unsigned int lay, unsigned int num);
            static std::string fact2string(FACT f);
    };  // class network




    /*******************************************/
    // neuron
    /*******************************************/

    /// <summary>
    /// Class neuron
    /// </summary>
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

        private:
            act x;                                  // Segnale in ingresso
            act y;                                  // Attività in uscita
            std::vector<synapse> syns;              // Sinapsi
            FACT fact;                              // Indice della funzione di attivazione
            act_func f_act;                         // Puntatori alla funzione di attivazione e...
            act_func f_act_der;                     // ...alla sua derivata.
            bool active = true;                     // Se false, non calcola né x dai pesi né y.
            bool input = false;                     // Se true: nodo di input, non calcola la x, solo la y, e abilita set_input
            
			#if TXT_INFO
            std::string name = "";
            std::string get_name() { return name; }
            void set_name(std::string s) { name = s; }
            #endif

        public:
            neuron();
			neuron(bool isInput);
            neuron(std::vector<neuron> &prev); 
            ~neuron();

            std::string to_string();

            bool get_active() {return active;}		// Neurone attivo / disattivo		
            void set_active(bool stat);
			
			bool get_input() { return input;}		// Neurone di input o standard
			void set_input(bool inp);				// Non modifica il vettore delle sinapsi
			
			FACT get_fact() {return fact;}			// Funzione di attivazione			
			std::string get_fact_name();
			void set_fact(FACT f);					// Cambia la funzione di attivazione, solo se non è un nodo di input

			act get_x() { return x; }				// Ingresso complessivo
			bool set_x(act x_in);                   // Modifica l'ingresso x, solo se è un nodo di input. Se no restituisce false.
			void calc_x();                          // Calcola x, solo se è active e se non è un nodo di input
			void calc_y();                          // Calcola y, solo se active




    };



    /*******************************************/
    // synapse
    /*******************************************/

    /// <summary>
    /// Class synapse
    /// </summary>
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
            std::string to_string();
            act x() {return w * pn->y;}
    };


}  // namespace neuro

#endif // NEURO_H
