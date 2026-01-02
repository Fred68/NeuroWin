

#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>
#include <memory>

namespace neuro
{
	class layer
	{
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
			inline layer(uint num, network &net, bool b) : _neurons(num, { net, b}) {};
			inline layer(uint num, network &net, layer &lay) : _neurons(num, { net, lay.get_neurons()}) {};

			inline neuron &operator[](uint i) { return _neurons[i]; }
			inline std::vector<neuron> &get_neurons() { return _neurons; }
			inline bool get_recalc_w() { return _recalc_w; }
			inline void set_recalc_w(bool recalc) { _recalc_w = recalc; }

			inline constexpr size_t size() { return _neurons.size(); }
			inline constexpr void push_back(const neuron& n) { _neurons.push_back(n); }

			inline Iterator begin() { return Iterator(_neurons.begin()); }
			inline Iterator end() { return Iterator(_neurons.end()); }

	};
}

#endif

