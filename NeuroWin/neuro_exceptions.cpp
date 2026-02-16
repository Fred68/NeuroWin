#include "neuro_exceptions.h"
#include <algorithm>        // for_each

namespace neuro
{

	neuro_exceptions::neuro_exception& neuro_exceptions::neuro_exception::operator=(neuro_exception const &other) noexcept
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


	const std::string neuro_exceptions::neuro_exception::what() const noexcept		// Nessun override di virtual const char* what() const noexcept
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

	const neuro_exceptions::neuro_exception &neuro_exceptions::create_exception(const neuro_exceptions::type type, bool is_error, std::string desc)
	{
		_exceptions.push_back(neuro_exceptions::neuro_exception(type, is_error, desc));
		return _exceptions.back();
	}


	bool neuro_exceptions::isOk()
	{
		uint count = std::count_if(_exceptions.begin(), _exceptions.end(), neuro_exception::is_ex_error);

		return (count == 0);
	}

	std::string neuro_exceptions::get_exceptions_string(bool show_warnings)
	{
		std::string ret, txt, txt_err, txt_warn;
		uint count, count_warn, count_err;
		bool err, warn;

		auto func_sel = [&](neuro_exception &ex)
			{
				if ((ex.is_error() && err) || (!ex.is_error() && warn))
				{
					txt += "\n" + ex.what();
					count++;
				}
			};

		count_err = count_warn = 0;

		txt = "", err = true, warn = false, count = 0;
		std::for_each(std::execution::seq, _exceptions.begin(), _exceptions.end(), func_sel);
		if (count > 0)		txt_err = txt;
		count_err = count;

		if (show_warnings)
		{
			txt = "", err = false, warn = true, count = 0;
			std::for_each(std::execution::seq, _exceptions.begin(), _exceptions.end(), func_sel);
			if (count > 0)	txt_warn = txt;
			count_warn = count;
		}

		if (count_err == 0)
		{
			ret += "network is ok";
		} else
		{
			ret += std::format("network has {0} errors:", count_err);
			ret += txt_err;
		}

		if (count_warn > 0)
		{
			ret += std::format("\nnetwork has {0} warnings:", count_warn);
			ret += txt_warn;
		}
		return ret;
	}


}