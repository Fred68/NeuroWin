



#include <iostream>

#include "neuro.h"


#include <vector>
#include <execution>
#include <algorithm>

#include <atomic>

//import modtest;

using namespace std;
using namespace neuro;
//using namespace pippospace;

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

    int sumR = std::reduce(std::execution::par, v.begin(), v.end(), j);
    // Usare questo: vd. https://en.cppreference.com/w/cpp/algorithm/reduce.html ???
    // No, perché l'operatore binario è solo su tipi std, non su classi
    int ssmR = std::reduce(std::execution::par, v.begin(), v.end(), j, [&](int tot, int x) {return tot + x; });
    std::cout << "sum (reduce)= " << sumR << std::endl;
    std::cout << "ssm (...+lambda op)= " << ssmR << std::endl;

    int sss = 0;
    auto func_s = [&](const int &x) {sss += x; };
    std::for_each(std::execution::par, v.begin(), v.end(), func_s); // Possibile errore per race condition
    std::cout << "sss (for_each on int, race condition?)" << sss << std::endl;

    atomic<int> sum_atm(0);
    auto func_atm = [&](const int &x) {sum_atm.fetch_add(x); };
    std::for_each(std::execution::par, v.begin(), v.end(), func_atm);
    std::cout << "sum_atm (for_each on atomic<int>)" << sum_atm << std::endl;

    //std::cout << "-----------------------------------------------\n";
    //std::cout << "module test" << std::endl;
    //std::cout << "-----------------------------------------------\n";
    //
    //pippo p(10);
    //cout << p.to_string() << endl;
    
    std::cout << "-----------------------------------------------\n";
    std::cout << "neuro test" << std::endl;
    std::cout << "-----------------------------------------------\n";

    std::vector<int> lays = {3,2,2};
    std::vector<FACT> facts ={FACT::sigmoid, FACT::sigmoid, FACT::sigmoid};
    init_data ini(lays,facts);
    std::cout << ini.to_string() << std::endl;

    network net(ini);          // Crea la rete
	vector<act> vinp = {0.1,0.2,0.9};
	vector<act> vout = {1,0};
	if(!net.prop_fw(vinp))	cout << "Error in fw propagation" << endl;
	if (!net.prop_bw(vout))	cout << "Error in bw propagation" << endl;

    std::cout << net.to_string();
    
    int x = getchar();

    return 0;
    
}

void print(vector<int> &v)
{
    std::cout << "[";
    for (int i = 0; i < v.size(); i++) { std::cout << v[i] << " "; }
    std::cout << "]\n";
}