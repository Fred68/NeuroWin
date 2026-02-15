#ifndef NEURO_EXC_H
#define NEURO_EXC_H

#include "neuro_exc_static.h"
#include <string>			// std::format
#include <chrono>			// high_resolution_clock
#include <vector>

namespace neuro
{
	class neuro_exceptions
	{
		public:
			_NEURO_EXC_ENUM;		// Usa la costante con l'enumerazione degli errori
			_NEURO_EXC_STR;			// Usa la costante con le stringhe statiche

		class neuro_exc
		{
			friend neuro_exceptions;

			private:

				type _type;
				bool _is_error;
				std::string _desc;
				std::chrono::system_clock::time_point _time;

				/// <summary>
				/// CTOR privato
				/// Visibile da neuro_exceptions
				/// </summary>
				/// <param name="type"></param>
				/// <param name="is_error"></param>
				/// <param name="desc"></param>
				inline neuro_exc(const type type = type::none, bool is_error = true, std::string desc = "") noexcept :
					_type(type), _is_error(is_error), _desc(desc), _time(std::chrono::system_clock::now()) {
				
				}

			public:
				
				/// <summary>
				/// Copy CTOR
				/// </summary>
				/// <param name="other"></param>
				inline neuro_exc(neuro_exc const &other)  noexcept :
					_type(other._type), _is_error(other._is_error), _desc(other._desc), _time(other._time) {
				}

				/// <summary>
				/// Assignment operator
				/// </summary>
				/// <param name="other"></param>
				/// <returns></returns>
				neuro_exc& operator=(neuro_exc const &other) noexcept;

				inline static bool is_ex_error(neuro_exc &nex) { return nex._is_error; }
				inline bool is_error() { return _is_error; }
				const std::string what() const noexcept;		// Nessun override di virtual const char* what() const noexcept

		};  // class neuro_exception

		private:
			std::vector<neuro_exc>	_exceptions;

		public:
			inline neuro_exceptions()
			{
				clear();			// Superfluo
			};
			
			inline void clear() {_exceptions.clear();}

			constexpr neuro_exc &create_exception(const neuro_exceptions::type type = neuro_exceptions::type::none, bool is_error = true, std::string desc = "");


	};


}
#endif
