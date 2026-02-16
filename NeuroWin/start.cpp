
#define INI_TEST false
#define SAVE_TEST true
#define LOAD_TEST true

#include <iostream>

#include "neuro_def.h"
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

// Prototyping
void learn(init_data &ini, learn_data &ld, neuro_exceptions &nexc);
void load(learn_data &ld, neuro_exceptions &nexc);

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
    
    std::cout << "-----------------------------------------------\n";
    std::cout << "neuro test" << std::endl;
    std::cout << "-----------------------------------------------\n";

	std::cout << get_build_time() << endl;


	std::cout << "-----------------------------------------------\n";
	std::cout << "inita data" << std::endl;
	std::cout << "-----------------------------------------------\n";
	
	std::vector<uint> lays = { 3, 5, 2 };
	std::vector<FACT> facts = { FACT::sigmoid, FACT::sigmoid, FACT::sigmoid };

	init_data ini(lays, facts, 0.05);									// init_data
	std::cout << "init_data:\n" << ini.to_string() << std::endl;

	neuro_exceptions excs;												// neuro_exceptions
	
	learn_data ld(ini.get_input_size(), ini.get_output_size());			// learn_data
	uint iInp, iOut;
	iOut = ld.add_output(vector<act>({ 1, 0 }));
	iInp = ld.add_input(vector<act>({ 0.1, 0.2, 0.9 }));
	ld.add_data(iInp, iOut, excs);
	iInp = ld.add_input(vector<act>({ 0.1, 0.1, 0.95 }));
	ld.add_data(iInp, iOut, excs);
	iInp = ld.add_input(vector<act>({ -0.1, 0.0, 0.8 }));
	ld.add_data(iInp, iOut, excs);

	iOut = ld.add_output(vector<act>({ 0, 1 }));
	iInp = ld.add_input(vector<act>({ 0.9, 0.2, 0.1 }));
	ld.add_data(iInp, iOut, excs);
	iInp = ld.add_input(vector<act>({ 0.85, 0.1, 0.0 }));
	ld.add_data(iInp, iOut, excs);
	iInp = ld.add_input(vector<act>({ 0.99, 0., 0.2 }));
	ld.add_data(iInp, iOut, excs);

	std::cout << "learn_data:\n" << ld.to_string(true) << std::endl;
	//for (auto it = ld.begin(); it != ld.end(); it++){}		// for(auto it : ld){} non è implementato

	#if false
	std::ofstream fs("test.bin", std::ios::binary);
	neuron x(*net);
	x.set_node_index(10);
	x.set_active(true);
	x.set_fact(neuro::FACT::relu);
	//x.write(fs);
	neuron nn = net->get_neuron(1,1);
	nn.write(fs);
	fs.close();

	std::ifstream fsr("test.bin", std::ios::binary);
	neuron y(*net);
	y.read(fsr);
	fsr.close();
	#endif

	std::cout << "-----------------------------------------------\n";
	std::cout << "[1 <enter>]\tLearn and save\n[2 <enter>]\tLoad and calc\n[0 <enter>]\tExit" << std::endl;
	std::cout << "-----------------------------------------------\n";
	

	char ch = '\0';
	
	{
		string chrs = "120X";
		while(chrs.find(ch)==std::string::npos)
		{
			ch = getchar();
		}
	}
	
	switch(ch)
	{
		case '1':
		{
			learn(ini,ld,excs);
		}
		break;
		case '2':
		{
			load(ld,excs);
		}
		break;
		default:
		break;
	}


	//net2.~network();		// Prova dtor
	//net->~network();

	std::cout << "\n-----------------------------------------------\n";
	std::cout << "end of test" << std::endl;
	std::cout << "-----------------------------------------------" << std::endl;


	//cout << learn_data::UINT_ERROR << endl;
	//cout << (uint) -1 << endl;
	getchar(), getchar();

	
    return 0;
    
}

void learn(init_data &ini, learn_data &ld, neuro_exceptions &nexc)
{
	#if SAVE_TEST

	std::shared_ptr<network> net = make_shared<network>(ini, nexc);		// Crea la rete, chiamando il ctor;
	try
	{
		neuro_exceptions::neuro_exception tmp = net->get_exceptions().create_exception(neuro_exceptions::pippo, false, "warning...");
		throw net->get_exceptions().create_exception(neuro_exceptions::pluto, true, "error...");
	} catch (std::exception const &ex)
	{
		cerr << "Catturata std::exception:\n" << ex.what() << std::endl;
	} catch (neuro_exceptions::neuro_exception const &nex)
	{
		cerr << "Catturata neuro::neuro_exception:\n" << nex.what() << std::endl;
	}

	/*try
	{
		_net->get_neuron(1, 0).set_fact(neuro::FACT::test_error);
	}
	catch (network::neuro_exception const &nex)
	{
		cerr << "Catturata neuro::neuro_exception:\n" << nex.what() << std::endl;
	}*/

	std::string ntok((net->isOk()) ? "ok" : "not ok");
	std::cout << "_net is " << ntok << std::endl;
	if (!net->isOk())
	{
		std::cout << net->get_exceptions().get_exceptions_string(false) << endl;
	}
	net->clear_exceptions();
	net->get_exceptions().create_exception(neuro_exceptions::pippo, false, "new warning...");
	std::cout << net->get_exceptions().get_exceptions_string(true) << endl;

	std::cout << "In: " << net->get_input_layer_size() << '\n' << "Out: " << net->get_output_layer_size() << endl;

	std::cout << ((ld.check_data_size(*net)) ? "learn data size ok" : "learn data size ok") << endl;

	uint cicli = 1;
	uint subcicli = 1;

	std::cout << "Cicli: ";
	cin >> cicli;
	std::cout << "Sottocicli: ";
	cin >> subcicli;

	std::cout << "\nnet before learning:\n" << net->to_string() << endl;

	std::chrono::milliseconds msec_elap(0);

	act errmed;

#if false
	cout << "Back-propagation single datum learning..." << endl;
	cout << ((net->backward_propagate(vinp, vout, cicli, errtot, msec_elap)) ? "ok" : "err") << '\n';
	cout << "Tempo: " << msec_elap << '\n';
	cout << "Err tot: " << errtot << '\n';
	cout << "\nFin:\n" << net->to_string() << endl;
#endif


	std::cout << "\n\nBack-propagation full data set learning..." << endl;

	net->backward_propagate(ld, cicli, subcicli, errmed, msec_elap);
	std::cout << "Tempo: " << msec_elap << '\n';
	std::cout << "Err med (quadratico): " << errmed << '\n';
	getchar();

	std::cout << "\nnet after learning:\n" << net->to_string() << endl;

	vector<act> vres(ini.get_output_size());

	std::cout << endl;
	std::cout << "--------------------------------------------------\n";
	std::cout << "Output della rete 'net' con forward-propagation\n";
	std::cout << "--------------------------------------------------\n";
	for (auto it = ld.begin(); it != ld.end(); it++)
	{
		auto vinp = it.get_input_v();
		auto vout = it.get_output_v();
		std::cout << "fw prop: " << ((net->forward_propagate(vinp, vres)) ? "ok" : "err") << endl;
		std::cout << "vinp (using): " << network::display_vector(vinp) << '\n';
		std::cout << "vout (obj.) : " << network::display_vector(vout) << '\n';
		std::cout << "vres (using): " << network::display_vector(vres) << endl;
		//	std::cout << "Net: " << net->to_string() << endl;
	}

	getchar();
	std::cout << "\n\nNet prima del calcolo di indici e topologia\n" << net->to_string() << endl;
	getchar();

	net->calc_indexes();
	net->get_topo();
	std::cout << "\n\nNet dopo il calcolo di indici e topologia\n" << net->to_string() << endl;
	getchar();


	std::cout << "\n\nSave neuron indexes:\n" << net->to_string() << endl;
	std::cout << "Salvataggio file..." << endl;
	net->save("pippo.bin");

	#endif
}

void load(learn_data &ld, neuro_exceptions &nexc)
{
	
	// TODO!!! Identificare errore dopo load(). fw_prop non fa nulla. Reimpostare puntatori a funzioni di attivazione ?

	#if LOAD_TEST

	network net(nexc);
	cout << "\nnet2 prima del caricamento:\n" << net.to_string() << endl;
	cout << "Caricamento file..." << endl;
	net.load("pippo.bin");
	cout << "\nnet2 dopo caricamento:\n" << net.to_string() << endl;

	//std::shared_ptr<learn_data> ld = make_shared<learn_data>(net);

	cout << endl;
	cout << "--------------------------------------------------\n";
	cout << "Output della rete 'net2' con forward-propagation\n";
	cout << "--------------------------------------------------\n";

	

	for (auto it = ld.begin(); it != ld.end(); it++)
	{
		auto vinp = it.get_input_v();
		auto vout = it.get_output_v();
		vector<act> vres;
		cout << "fw prop: " << ((net.forward_propagate(vinp, vres)) ? "ok" : "err") << endl;
		cout << "vinp (using): " << network::display_vector(vinp) << '\n';
		cout << "vout (obj.) : " << network::display_vector(vout) << '\n';
		cout << "vres (using): " << network::display_vector(vres) << endl;
		cout << "Net2: " << net.to_string() << endl;
	}

	#endif
}

#undef INI_TEST
