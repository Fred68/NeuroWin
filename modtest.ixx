
export module modtest;

#include <string>
#include <format>

namespace pippospace
{
	export class pippo
	{
		private:
			int _p;

		public:
			pippo() : pippo(0) {}
			pippo(int p) : _p(p) {}
			std::string to_string();
	};


	std::string pippo::to_string()
	{
		return std::format("p={0}",_p);
	}
}