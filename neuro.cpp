

/***********************************************/
// neuro.cpp
// Implementation for neural network components
// Standard C++ 20.0
// Version 0.1
/***********************************************/



#if _DEBUG
    #include <iostream>
    #define _DEBUG_NEURO_DET true
#endif

#include <format>
#include <memory>
#include <vector>

#include "neuro.h"


using namespace std;

namespace neuro
{  
    /*******************************************/
    // network
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
        for(int i=0; i < _nLays; i++)       // create layers
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
        
        name_elements();

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

        for (int i=0; i < _nLays; i++)
        {
            txt += std::format("Layer: {0}\n", i);
            for(neuron n : layers[i])
            {
                txt += std::format("{0}\n", n.to_string());
            }
        }
        return txt;
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

    /*******************************************/
    // neuron
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
            //synapse s(n,0);
            //syns.push_back(s);
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
        string txt = std::format("x={0}, y={1}",x,y);
        for(synapse s : syns)
        {
            txt = txt + std::format("[{0},{1}]",s.pn->get_name(),s.w);
        }
        #if TXT_INFO
        txt = name + ": " + txt;
        #endif
        return txt;
    }



    /*******************************************/
    // synapse
    /*******************************************/

    synapse::synapse()
    {
        pn = std::make_shared<neuron>();        // Alloca un neurone vuoto e ne crea il puntatore
        w = (act)0;
        #if _DEBUG_NEURO_DET
        cout << "synapse()\n";
        #endif
    }

    synapse::synapse(neuron &p_n, act ws)
    {
       // pn = std::make_shared<neuron>(p_n);     // No: make_shared crea una copia
        pn = std::shared_ptr<neuron>(&p_n);
        w = ws;
        pn->set_name(pn->get_name()+"-");
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