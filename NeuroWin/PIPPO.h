#ifndef PIPPO_H
#define PIPPO_H

// Test. Escluso dalla compilazione
#if false
namespace neuro
{
	class network;

	class PIPPO
	{
		private:
			int _i;
			const std::shared_ptr<network> _pnet;

		public:
			PIPPO(network *net, int i = 0) : _pnet(net), _i(i){};
			void call()
			{
				//_pnet->get_reference();
				/*(network&)_net->get_reference();*/
			}
	};

}
#endif

#endif