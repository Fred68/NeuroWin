

/*************************************************/
/* neuro.cpp                                     */
/* Implementation for neural network components  */
/* Standard C++ 20.0                             */
/* Version 0.1                                   */
/* Copyright FcSoft                              */
/*************************************************/



#ifndef NEURO_H
#define NEURO_H

#define ACT_DBL                     // Definizione del tipo di dato per l'attività: double 
//#undef ACT_DBL                      // Se non definito: float

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
    
    enum class FACT { sigmoid = 0, tanh, relu, one, Count };

    // Forward declarations
    class neuron;
    class synapse;



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
            std::vector<std::vector<neuron>> layers;

        public:
            network();
            network(std::vector<int> nlay);
            ~network();
            std::string to_string();
            neuron& get_neuron(unsigned int lay, unsigned int num);

        private:
            neuron& get_at(int lay, int num) {return (layers[lay])[num];}
            #if TXT_INFO
            void name_elements();
            #endif

    };  // class network




    /*******************************************/
    // neuron
    /*******************************************/

    /// <summary>
    /// Class neuron
    /// </summary>
    class neuron
    {

        friend class network;
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
        static FACT fact_default() {return FACT::tanh;}

        private:
            act x;                                  // Segnale in ingresso
            act y;                                  // Attività in uscita
            std::vector<synapse> syns;              // Sinapsi
            FACT fact;                              // Indice della funzione di attivazione
            act_func f_act;                         // Puntatori alla funzione di attivatore e...
            act_func f_act_der;                     // ...alla sua derivata.

            #if TXT_INFO
            std::string name = "";
            std::string get_name() { return name; }
            void set_name(std::string s) { name = s; }
            #endif
            

            FACT get_fact();
            void set_fact(FACT f);
            std::string get_fact_name();

            // Calcolo ingresso complessivo
            void calc_x();
            void calc_y();

        public:
            
            neuron();
            neuron(std::vector<neuron> &prev);           
            ~neuron();
            std::string to_string();
            
            
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
