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

#define TXT_INFO true               // Informazioni aggiuntive in nodi e sinapsi

#include <string>
#include <vector>
#include <execution>
#include <algorithm>
#include <memory>

// USARE std::for_each(std::execution::par, data.begin(), data.end(), [](int&) {std::cout << "Hello, World!" << std::endl;});


namespace neuro
{

    // @brief Tipo di dato per l'attività neurale: act
    #ifdef ACT_DBL
    typedef double act;
    #else
    typedef float act;
    #endif
    
    // Forward declarations
    class neuron;
    class synapse;

   

    /// <summary>
    /// Class neuron
    /// </summary>
    class neuron
    {
        private:
            act x;                                  // Segnale in ingresso
            act y;                                  // Attività in uscita
            std::vector<synapse> syns;              // Sinapsi
            #if TXT_INFO
            std::string name = "";
            std::string get_name() { return name; }
            void set_name(std::string s) { name = s; }
            #endif


        public:
            neuron();
            neuron(std::vector<neuron> &prev);           
            ~neuron();

            std::string to_string();
            
        friend class network;
        friend class synapse;
    };

    /// <summary>
    /// Class synapse
    /// </summary>
    class synapse
    {
        private:
            std::shared_ptr<neuron> pn;
            //neuron *pn;
            act    w;

        public:
            synapse();
            synapse(neuron &p_n, act ws);
            ~synapse();
            
            std::string to_string();
        
        friend class neuron;
    };

    // @brief Class network
    class network
    {
        private:
            int _nLays = 0;
            std::vector<std::vector<neuron>> layers;
            neuron& get_at(int lay, int num)
            {
                return (layers[lay])[num];
            }
            
        public:
            network();
            network(std::vector<int> nlay);
            ~network();

            std::string to_string();

            neuron& get_neuron(int lay, int num)
            {
                if( (lay<0) || (lay>=_nLays))
                    throw new std::exception("Layer out of range");
                else if((num < 0) || ( num >= layers[lay].size())) 
                    throw new std::exception("Node out of range");
                 else
                    return get_at(lay,num);
            }
            

        private:
            #if TXT_INFO
            void name_elements();
            #endif

    };  // class network


}  // namespace neuro
#endif // NEURO_H
