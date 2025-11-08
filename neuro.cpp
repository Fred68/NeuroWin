

/***********************************************/
// neuro.cpp
// Implementation for neural network components
// Standard C++ 20.0
// Version 0.1
/***********************************************/





#include <format>
#include <memory>
#include <vector>
#include <cmath>

#include <execution>
#include <algorithm>


#include "neuro.h"


using namespace std;

namespace neuro
{  
    /*******************************************/
    //
    // network
    //
    /*******************************************/

    network::network()
    {
        _nLays = 0;
        #if _DEBUG
        cout << "network()\n";
        #endif
    }
    
    network::network(std::vector<int> nlay)
    {
        _nLays = nlay.size();
        for(unsigned int i=0; i < _nLays; i++)       // create layers
        {
            vector<neuron> *vn;
            if(i==0)
            {
                vn = new vector<neuron>(nlay[i]);     // allocate vector<neuron> with empty ctor for neuron
            }
            else
            {
                vn = new vector<neuron>(nlay[i],layers[i-1]);   // allocate vector<neuron>
            }
            layers.push_back(*vn);
        }
        
        #if TXT_INFO
        name_elements();
        #endif
        
        #if _DEBUG
        cout << "network(" << _nLays <<")\n";
        #endif

       
    }

    network::~network()
    {
        _nLays = 0;

        #if _DEBUG
        cout << "~network()\n";
        getchar();
        #endif
    }

    std::string network::to_string()
    {
        string txt;
        txt += std::format("Layers: {0}\n", _nLays);

        for (unsigned int i=0; i < _nLays; i++)
        {
            txt += std::format("Layer: {0}\n", i);
            for(neuron n : layers[i])
            {
                txt += std::format("{0}\n", n.to_string());
            }
        }
        return txt;
    }
    
    neuron& network::get_neuron(unsigned int lay, unsigned int num)
    {
        if (lay >= _nLays)
            throw new std::exception("Layer out of range");
        else if (num >= layers[lay].size())
            throw new std::exception("Node out of range");
        else
            return get_at(lay, num);
    }

    #if TXT_INFO
    void network::name_elements()
    {
        for (int i = 0; i < _nLays; i++)
            for (int j = 0; j < layers[i].size(); j++)
            {
                get_at(i,j).set_name(std::format("L{0}N{1}",i,j));
            }
    }
    #endif



//-----------------------------------------------------------

    /*******************************************/
    //
    // neuron
    //
    /*******************************************/

    neuron::neuron()
    {
        x = y = 0;
        #if _DEBUG_NEURO_DET
        cout << "neuron()\n";
        #endif
    }

    neuron::neuron(std::vector<neuron> &prev)
    {
        x = y = 0;
        for(neuron &n : prev)
        {
            syns.push_back(synapse(n,0));
        }
        #if _DEBUG_NEURO_DET
        cout << "neuron(neuron &prev)\n";
        #endif
    }

    neuron::~neuron()
    {
        #if _DEBUG_NEURO_DET
        cout << "~neuron()\n";
        getchar();
        #endif
    }

    std::string neuron::to_string()
    {
        string txt = std::format("x={0},y={1}",x,y);
        for(synapse s : syns)
        {
            if(s.pn!=nullptr)
            {
                string nn = "";
                #if TXT_INFO
                nn = s.pn->get_name()+",";
                #endif
                txt = txt + std::format("[{0}{1}]", nn, s.w);
            }
        }
        #if TXT_INFO
        txt = name + ": " + txt;
        #endif
        return txt;
    }
    
    void neuron::calc_x()
    {
        
        // USARE std::for_each(std::execution::par, data.begin(), data.end(), [](int&) {std::cout << "Hello, World!" << std::endl;});
        // std::for_each(std::execution::seq,syns.begin(),syns.end(),[](synapse&){}


        // act tot = std::accumulate(std::execution::par,syns.begin(),syns.end(),0.0);
    }
    /*******************************************/
    // Funzioni di attivazione
    act neuron::sigmoid(neuron &n)
    {
        #ifdef ACT_DBL
            return 1.0 / (1.0 + std::exp(-n.x));
        #else
            return 1.0f / (1.0f + std::expf(-n.x));
        #endif
    }
    act neuron::sigmoid_der(neuron &n)
    {
        #ifdef ACT_DBL
            return n.y * (1.0 - n.y);
        #else
            return n.y * (1.0f - n.y);
        #endif    
    }
    act neuron::hyptangent(neuron &n)
    {
        #ifdef ACT_DBL
            return std::tanh(n.x);
        #else
            return std::tanhf(n.x);
        #endif
    }
    act neuron::hyptangent_der(neuron &n)
    {
        #ifdef ACT_DBL
            return 1.0 - n.y * n.y;
        #else
            return 1.0f - n.y * n.y;
        #endif    
    }
    act neuron::relu(neuron &n)
    {
        #ifdef ACT_DBL
            return (n.x > 0) ? n.x : 0.0;
        #else
            return (n.x > 0) ? n.x : 0.0f;
        #endif
    }
    act neuron::relu_der(neuron &n)
    {
        #ifdef ACT_DBL
            return (n.x > 0) ? 1.0 : 0.0;
        #else
            return (n.x > 0) ? 1.0f : 0.0f;
        #endif    
    }

//-----------------------------------------------------------

    /*******************************************/
    //
    // synapse
    //
    /*******************************************/

    synapse::synapse()
    {
        pn = std::shared_ptr<neuron>(nullptr);  // Non usa pn=std::make_shared<neuron>() perché alloca un nuovo oggetto
        w = (act)0;
        #if _DEBUG_NEURO_DET
        cout << "synapse()\n";
        #endif
    }

    synapse::synapse(neuron &p_n, act ws)
    {
        pn = std::shared_ptr<neuron>(&p_n);     // Non usa pn=std::make_shared<neuron>(p_n) perché crea una copia
        w = ws;
        #if _DEBUG_NEURO_DET
        cout << "synapse(p_n)\n";
        #endif
    }

    synapse::~synapse()
    {
        #if _DEBUG_NEURO_DET
        cout << "~synapse()\n";
        getchar();
        #endif
    }
} 