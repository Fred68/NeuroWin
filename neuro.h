/***********************************************/
// neuro.h
// Header file for neural network components
// // Standard C++ 20.0
// Version 0.1
/***********************************************/



#ifndef NEURO_H
#define NEURO_H

#define ACT_DBL                     // Definizione del tipo di dato per l'attività: double 
#undef ACT_DBL                      // Se non definito: float

#define TXT_INFO false              // Informazioni aggiuntive in nodi e sinapsi

#if _DEBUG
    #include <iostream>
    #define _DEBUG_NEURO_DET false
#endif

#include <string>
#include <vector>
#include <memory>



namespace neuro
{

    // Tipo di dato per l'attività neurale: act
    #ifdef ACT_DBL
    typedef double act;
    #else
    typedef float act;
    #endif
    
    // Forward declarations
    class neuron;
    class synapse;



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

        private:
            act x;                                  // Segnale in ingresso
            act y;                                  // Attività in uscita
            std::vector<synapse> syns;              // Sinapsi
            #if TXT_INFO
            std::string name = "";
            std::string get_name() { return name; }
            void set_name(std::string s) { name = s; }
            #endif
            
            // Funzioni di attivazione
            act sigmoid(neuron &n);
            act sigmoid_der(neuron &n);
            act hyptangent(neuron &n);
            act hyptangent_der(neuron &n);
            act relu(neuron &n);
            act relu_der(neuron &n);

            // Calcolo neuron
            void calc_x();

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


}  // namespace neuro
#endif // NEURO_H
