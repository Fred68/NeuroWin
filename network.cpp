

#include "neuro.h"


namespace neuro
{
    /*******************************************/
    /*                                         */
    /* network                                 */
    /*                                         */
    /*******************************************/

    network::network()
    {
        _nLays = 0;
        #if _DEBUG
        std::cout << "network()\n";
        #endif
    }

    network::network(init_data &ini_data)
    {
        _nLays = std::min( (unsigned int) ini_data._layers.size(), (unsigned int)ini_data._types.size() );

        for(unsigned int i=0; i < _nLays; i++)		// Crea i livelli, ognuno con un nodo in più (uscita 1, per i bias)
        {
            std::vector<neuron> *vn;				// Alloca il vettore con vector<T>(numero, parametri per il ctor di T).
            if(i==0)								// Per il primo livello, crea neuroni di input, usando come ctor:
            {										// ...neuron(bool true) 
                vn = new std::vector<neuron>(ini_data._layers[i] + 1,true);    
            }
            else
            {										// Per gli altri, usa neuron()   
                vn = new std::vector<neuron>(ini_data._layers[i] + 1,_layers[i-1]);   // allocate vector<neuron>
            }
            _layers.push_back(*vn);

            int jmax = (unsigned int) _layers.back().size();
            for(int j=0; j<jmax; j++)
            {
                if(j == jmax-1)
                {
                    _layers.back()[j].set_fact(FACT::one);           // Nodo aggiunto: uscita sempre a 1
                    _layers.back()[j].calc_y();                      // Calcola l'uscita e...
                    _layers.back()[j].set_active(false);             // ...disattiva
                }
                else
                {
                    _layers.back()[j].set_fact(ini_data._types[i]);
                }
            }
        }
        
        #if TXT_INFO
        name_elements();
        #endif
        
        #if _DEBUG
        std::cout << "network(" << _nLays <<")\n";
        #endif

       
    }
    network::~network()
    {
        _nLays = 0;

        #if _DEBUG
        std::cout << "~network()\n";
        int x = getchar();
        #endif
    }

    std::string network::fact2string(FACT f)
    {
        std::string str = "";
        switch(f)
        {
            case FACT::one:
                str = "one";
                break;
            case FACT::sigmoid:
                str = "sigmoid";
                break;
            case FACT::tanh:
                str = "tanh";
                break;
            case FACT::relu:
                str = "relu";
                break;
            case FACT::id:
                str = "id";
                break;
            default:
                str = "FACT error";
                break;
        }
        return str;
    }
    std::string network::to_string()
    {
        std::string txt;
        txt += std::format("Layers: {0}\n", _nLays);

        for (unsigned int i=0; i < _nLays; i++)
        {
            txt += std::format("Layer: {0}\n", i);
            for(neuron n : _layers[i])
            {
                txt += format("{0}\n", n.to_string());
            }
        }
        return txt;
    }
    neuron& network::get_neuron(unsigned int lay, unsigned int num)
    {
        if (lay >= _nLays)
            throw new std::exception("Layer out of range");
        else if (num >= _layers[lay].size())
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




}