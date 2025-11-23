#pragma once

#include "neuro.h"

namespace neuro
{
    /*******************************************/
    // test
    /*******************************************/

    class test
    {
    private:
        int _x;
        int _y;
    public:
        test() : test(0, 0) {}
        test(int x, int y) : _x(x), _y(y) {}
        std::string to_string();
    };
    class test1
    {
    private:
        test x;
        test *y;
        test z;
        // test w(3,1); // No, inizializzare nel ctor

        test1() : y(new test(1, 2)), z(3, 4) {}
        ~test1() { delete y; }
    };

}