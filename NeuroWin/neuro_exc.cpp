#include "neuro_exc.h"

namespace neuro
{

	neuro_exceptions::neuro_exc& neuro_exceptions::neuro_exc::operator=(neuro_exc const &other) noexcept
	{
		if (this != &other)
		{
			_type = other._type;
			_is_error = other._is_error;
			_desc = other._desc;
			_time = other._time;
		}
		return *this;
	}


	const std::string neuro_exceptions::neuro_exc::what() const noexcept		// Nessun override di virtual const char* what() const noexcept
	{
		std::string txt;
		txt = "[";
		std::string x = std::format("{0}", _time);
		txt += x;
		txt += "] ";
		txt += _str[_type];
		if (_desc.size() > 0)	txt += " : " + _desc;
		return txt;
	}

	constexpr neuro_exceptions::neuro_exc &neuro_exceptions::create_exception(const neuro_exceptions::type type, bool is_error, std::string desc)
	{
		_exceptions.push_back(neuro_exceptions::neuro_exc(type, is_error, desc));
		return _exceptions.back();
	}


}