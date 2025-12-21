
#define INI_TEST false


#include <iostream>
//#include <tuple>			// learn_data iterator

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
	std::shared_ptr<network> net;
	std::shared_ptr<learn_data> ld;
	try
	{
		net = make_unique<network>(ini);
		ld = make_unique<learn_data>(net);		
	}
	catch(std::exception const &ex)
	{
		cerr << ex.what() << std::endl;
	}
	
	cout << "In: " << net->get_input_layer_sz() << '\n' << "Out: " << net->get_output_layer_sz() << endl;
	
	//learn_data ld(net);
	
	uint iInp, iOut;

	iOut = ld->add_output(vector<act>({ 1, 0 }));
	iInp = ld->add_input(vector<act>({ 0.1, 0.2, 0.9 }));
	ld->add_data(iInp, iOut);
	iInp = ld->add_input(vector<act>({ 0.1, 0.1, 0.95 }));
	ld->add_data(iInp, iOut);
	iInp = ld->add_input(vector<act>({ -0.1, 0.0, 0.8 }));
	ld->add_data(iInp, iOut);

	iOut = ld->add_output(vector<act>({ 0, 1 }));
	iInp = ld->add_input(vector<act>({ 0.9, 0.2, 0.1 }));
	ld->add_data(iInp, iOut);
	iInp = ld->add_input(vector<act>({ 0.85, 0.1, 0.0 }));
	ld->add_data(iInp, iOut);
	iInp = ld->add_input(vector<act>({ 0.99, 0., 0.2 }));
	ld->add_data(iInp, iOut);

	cout << "learn_data iterator:\n";
	for(auto it = ld->begin(); it != ld->end(); it++)		// for(auto it : ld){} non è implementato
	{
		cout << network::display_vector(it.get_input_v()) << " -> " << network::display_vector(it.get_output_v()) << endl;
	}
	
	cout << ((ld->check_data_size()) ? "learn data size ok" : "learn data size ok") << endl;

	uint cicli = 1;
	uint subcicli = 1;

	cout << "Cicli: ";
	cin >> cicli;
	cout << "Sottocicli: ";
	cin >> subcicli;

	//vector<act> vinp = std::get<0>(ld->get_data(0));
	//vector<act> vout = std::get<1>(ld->get_data(0));
	//cout << "vinp (teach.): " << network::display_vector(vinp) << '\n';
	//cout << "vout (teach.): " << network::display_vector(vout) << endl;

	cout << "\nnet before learning:\n" << net->to_string() << endl;


	std::chrono::milliseconds msec_elap(0);
	
	act errtot,errmed;
	
	#if false
	cout << "Back-propagation single datum learning..." << endl;
	cout << ((net->backward_propagate(vinp, vout, cicli, errtot, msec_elap)) ? "ok" : "err") << '\n';
	cout << "Tempo: " << msec_elap << '\n';
	cout << "Err tot: " << errtot << '\n'; 
	cout << "\nFin:\n" << net->to_string() << endl;
	#endif


	cout << "\n\nBack-propagation full data set learning..." << endl;
	
	net->backward_propagate(ld, cicli, subcicli, errmed, msec_elap);
	cout << "Tempo: " << msec_elap << '\n';
	cout << "Err med: " << errmed << '\n';
	cout << "\nFin:\n" << net->to_string() << endl;

	cout << "\nnet after learning:\n" << net->to_string() << endl;

	vector<act> vres(2);
	cout << "Forward-propagation:" << endl;

	for (auto it = ld->begin(); it != ld->end(); it++)
	{
		auto vinp = it.get_input_v();
		auto vout = it.get_output_v();
		cout << ((net->forward_propagate(vinp, vres)) ? "ok" : "err") << endl;
		cout << "vinp (using): " << network::display_vector(vinp) << '\n';
		cout << "vout (obj.) : " << network::display_vector(vout) << '\n';
		cout << "vres (using): " << network::display_vector(vres) << endl;
	}

	//cout << ((net->forward_propagate(vinp, vres)) ? "ok" : "err") << endl;
	//cout << "vinp (using): " << network::display_vector(vinp) << '\n';
	//cout << "vout (obj.) : " << network::display_vector(vout) << '\n';
	//cout << "vres (using): " << network::display_vector(vres) << endl;


	std::cout << "\n-----------------------------------------------\n";
	std::cout << "end of test" << std::endl;
	std::cout << "-----------------------------------------------\n";

	//cout << learn_data::UINT_ERROR << endl;
	//cout << (uint) -1 << endl;
	getchar();
	getchar();

    return 0;
    
}



#undef INI_TEST
