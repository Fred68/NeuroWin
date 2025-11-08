
#include <iostream>


#include "neuro.h"


#include <vector>
#include <execution>
#include <algorithm>



using namespace std;
using namespace neuro;

// Function prototyping
void print(vector<int> &v);

int main()
{
    int j = 0;
    auto func_x2 = [&](int &x) {x = x * 2; };
   

    std::vector<int> v = {1,2,3,4,5};
    print(v);
        
    std::for_each(v.begin(),v.end(), func_x2);
    print(v);
    
    int sum = std::accumulate(v.begin(), v.end(), j);   // Non ha versione parallela
    int ssm = std::accumulate(v.begin(), v.end(), j, [&](int tot, int x){return tot + x;});
    std::cout << "sum (accumulate)= " << sum << std::endl;
    std::cout << "ssm (...+lambda op)= " << ssm << std::endl;

    int sss = 0;
    auto func_s = [&](const int &x) {sss += x; };
    std::for_each(std::execution::par, v.begin(), v.end(), func_s); // Possibile errore per race condition
    std::cout << "sss " << sss << std::endl;

    int sumR = std::reduce(std::execution::par, v.begin(), v.end(), j);
    // Usare questo: vd. https://en.cppreference.com/w/cpp/algorithm/reduce.html
    int ssmR = std::reduce(std::execution::par, v.begin(), v.end(), j, [&](int tot, int x) {return tot + x; });
    std::cout << "sum (reduce)= " << sumR << std::endl;
    std::cout << "ssm (...+lambda op)= " << ssmR << std::endl;

    std::cout << "neuro test" << std::endl;
    
    std::vector<int> lays = {3,2,2};

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

void print(vector<int> &v)
{
    std::cout << "[";
    for (int i = 0; i < v.size(); i++) { std::cout << v[i] << " "; }
    std::cout << "]\n";
}