
#define INI_TEST false


#include <iostream>

#include "network.h"

#if INI_TEST
#include <vector>
#include <execution>
#include <algorithm>
#include <atomic>
#endif


//import modtest;

using namespace std;
using namespace neuro;

//using namespace pippospace;

int main()
{
	#if INI_TEST
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
	#endif

    //std::cout << "-----------------------------------------------\n";
    //std::cout << "module test" << std::endl;
    //std::cout << "-----------------------------------------------\n";
    //
    //pippo p(10);
    //cout << p.to_string() << endl;
    
    std::cout << "-----------------------------------------------\n";
    std::cout << "neuro test" << std::endl;
    std::cout << "-----------------------------------------------\n";

	//vector<int> pippo(0);

	// Ini
    std::vector<int> lays = {3, 5, 2};
    std::vector<FACT> facts ={FACT::sigmoid, FACT::sigmoid, FACT::sigmoid};
    
	init_data ini(lays,facts,0.05);
    std::cout << "init_data:\n" << ini.to_string() << std::endl;
	
	// net
	std::unique_ptr<network> net;
	try
	{
		net = make_unique<network>(ini);          // Crea la rete
	}
	catch(std::exception const &ex)
	{
		cout << ex.what() << std::endl;
	}
	
	cout << "In: " << net->get_input_layer_sz() << "\n" << "Out: " << net->get_output_layer_sz() << endl;
	uint cicli = 1;
	cout << "Cicli: ";
	cin >> cicli;

	// data
	vector<act> vinp = {0.1,0.2,0.9};
	vector<act> vout = {1,0};
	vector<act> vres(2);

	cout << "vinp (teach.): " << network::display_vector(vinp) << '\n';
	cout << "vout (teach.): " << network::display_vector(vout) << endl;

	cout << "\nIni:\n" << net->to_string() << endl;

	std::chrono::milliseconds msec_elap(0);

	cout << "Back-propagation learning..." << endl;
	cout << ((net->backward_propagate(vinp, vout, cicli, msec_elap)) ? "ok" : "err") << '\n';
	cout << "Tempo: " << msec_elap << endl;
	cout << "\nFin:\n" << net->to_string() << endl;

	cout << "Forward-propagation (test):" << endl;
	cout << ((net->forward_propagate(vinp, vres)) ? "ok" : "err") << endl;
	cout << "vinp (using): " << network::display_vector(vinp) << '\n';
	cout << "vout (obj.) : " << network::display_vector(vout) << '\n';
	cout << "vres (using): " << network::display_vector(vres) << endl;


	getchar();
	getchar();

	//auto v = std::ranges::iota_view((uint)0, (uint)5);
	//cout << "iota sz: "<< v.size() << endl;
	//for_each(v.begin(),v.end(),[&](uint i){cout << i << endl;});
	//x = getchar();

    return 0;
    
}



#undef INI_TEST
