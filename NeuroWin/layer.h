

#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>
#include <memory>
#include <fstream>			// I/O su stream (binario)

namespace neuro
{
	class layer
	{

		network &_net;
		std::vector<neuron> _neurons;	/// Vettore con i neuroni. Classe derivata: più complessa da scrivere.
		bool _recalc_w = true;			/// Ricalcola i pesi con back propagation. Se false: bloccati
		
		struct Iterator
		{
			using iterator_category = std::forward_iterator_tag;
			using value_type = neuron;
			using element_type = neuron;
			using reference = neuron&;
			using difference_type = std::ptrdiff_t;

			std::vector<neuron>::iterator _it;

			public:
				inline Iterator(std::vector<neuron>::iterator iterator) : _it(iterator) {}
				inline Iterator& operator++() { ++_it; return *this; }
				inline bool operator!=(const Iterator& iter) const { return _it != iter._it; }
				inline bool operator==(const Iterator& iter) const { return _it == iter._it; }
				inline reference operator*() const {return *_it;}
		};

		public:
			/// <summary>
			/// Ctor (semplice)
			/// </summary>
			/// <param name="net"></param>
			inline layer(network &net) : _neurons(0, {net, true}), _net(net) {};

			#if _COPY_CTORS_
			layer(const layer& other);
			layer& operator=(const layer& other);
			#endif
			#if _MOVE_CTORS_
			layer(const layer&& other);
			layer& operator=(const layer&& other);
			#endif

			/// <summary>
			/// Ctor
			/// Chiama il costruttore di vector<neuron> passando il numero di elementi e gli argomenti per il ctor di neuron
			/// </summary>
			/// <param name="num">numero di neuroni</param>
			/// <param name="net">rif. alla rete</param>
			/// <param name="b">livello di input?</param>
			inline layer(uint num, network &net, bool b) : _neurons(num, { net, b}), _net(net) {};

			/// <summary>
			/// Ctor
			/// Chiama il costruttore di vector<neuron> passando il numero di elementi e gli argomenti per il ctor di neuron
			/// </summary>
			/// <param name="num">numero di neuroni</param>
			/// <param name="net">rif. alla rete</param>
			/// <param name="lay">rif. al livello precedente</param>
			inline layer(uint num, network &net, layer &lay) : _neurons(num, { net, lay.get_neurons()}), _net(net) {};




			inline neuron &operator[](uint i) { return _neurons[i]; }
			inline std::vector<neuron> &get_neurons() { return _neurons; }
			inline bool get_recalc_w() { return _recalc_w; }
			inline void set_recalc_w(bool recalc) { _recalc_w = recalc; }

			inline constexpr size_t size() { return _neurons.size(); }
			inline constexpr void push_back(const neuron& n) { _neurons.push_back(n); }

			inline Iterator begin() { return Iterator(_neurons.begin()); }
			inline Iterator end() { return Iterator(_neurons.end()); }

			void write(std::ofstream &fs);
			void read(std::ifstream &fs);
	};
}

#endif

