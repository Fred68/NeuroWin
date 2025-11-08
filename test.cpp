
#include <iostream>
#include "neuro.h"

using namespace neuro;

int main()
{
    // act x = 0.5;
    std::cout << "neuro test" << std::endl;
    
    std::vector<int> lays = {1,1,1};

    network net(lays);
    
    std::cout << net.to_string();
    
    //  std::cout << net.f_sigmoid(x) << std::endl;
    //std::cout << f_sigmoid(x) << std::endl;
    // std::cout << f_tanh(x) << std::endl;
    /*int x;
    std::cin >> x;*/

    getchar();

    return 0;
    
}